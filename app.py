from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import numpy as np
import sys
import secrets


app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)  # Generate a new secret key each time the server restarts

# Add your YOLOv5 directory to sys.path
sys.path.append('/home/saad/Desktop/AI_Project/yolov5')

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Define device
device = select_device('')  # '' for CPU, '0' for CUDA

# Define a dictionary for item prices
item_prices = {
    'bowl': 3.50, 
    'spoon': 1.25, 
    'cup': 2.00,    
    'plate': 2.75   
}

# Load the model
model = attempt_load('model/best.pt', device=device)
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # adjust img size to model stride

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

def run_detection(image_path):
    img0 = Image.open(image_path).convert('RGB')
    img_np = np.array(img0)
    img = letterbox(img_np, new_shape=imgsz, stride=stride)[0]
    img = np.array(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detections = []
    annotator = Annotator(img0, line_width=3, example=str(model.names))
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.size).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                item_name = model.names[c]
                label = f'{item_name} {conf:.2f}'
                price = item_prices.get(item_name.lower(), 0)  # Get price from dictionary, default to 0 if not found
                detections.append((item_name, f"{conf:.2f}", price))
                annotator.box_label(xyxy, label, color=colors(c, True))
        else:
            print('No detections')

    save_filename = os.path.basename(image_path)
    if not save_filename.endswith('_detected.jpg'):
        save_filename += '_detected.jpg'
    save_path = os.path.join('static/uploads', save_filename)
    img0.save(save_path)
    return detections



@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()  # Ensure any session data is cleared when accessing the home page
    return render_template('index.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    session.clear()  # Ensure any session data is cleared when accessing the home page
    return render_template('report.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)

            # Run detection which appends '_detected.jpg' to filenames
            detections = run_detection(filepath)  # This function also saves the detected file
            
            # Ensuring 'items' is in session and extending it with new detections
            if 'items' not in session:
                session['items'] = []
            session['items'].extend(detections)
            session.modified = True

            return redirect(url_for('detect'))

    return render_template('detection.html', items=session.get('items', []))


@app.route('/preview/<path:filename>')
def preview(filename):
    # Ensure the filename ends with '_detected.jpg'

    filename = filename + '.jpg'
    if not filename.endswith('_detected.jpg'):
        filename += '_detected.jpg'

    # Extract the base item name (e.g., "bowl" from "bowl.jpg")
    item_name = filename.split('.')[0]

    # Construct the full static path
    image_path = f'uploads/{filename}'

    return render_template('Preview.html', item_name=item_name, image_path=image_path)


@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
