# SnapBill AI

SnapBill AI is an AI-powered system designed to streamline the billing process for kitchen crockery items using YOLOv5 for object detection and a web-based interface for easy interaction. This project combines computer vision with web technologies to automate the identification of crockery items and generate bills efficiently.

## Features

- **Object Detection**: Utilizes YOLOv5 to accurately detect and identify various kitchen crockery items.
- **Automated Billing**: Automatically generates bills based on detected items.
- **Web-Based Interface**: Easy-to-use interface for interacting with the system.
- **Efficiency**: Reduces manual effort and time in the billing process.

## Technologies Used

- **YOLOv5**: For object detection and identification of crockery items.
- **Web Technologies**: HTML, CSS, JavaScript for the frontend interface.
- **Backend**: Python, Flask for server-side processing.
- **Database**: MySQL/PostgreSQL for storing item data and billing information.

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/SaadGillani7/snapbill-ai.git
    cd snapbill-ai
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Database**:
    - Set up your database (MySQL/PostgreSQL).
    - Update the database configuration in the settings file.


4. **Start the Server**:
    ```bash
    python manage.py runserver
    ```

5. **Access the Web Interface**:
    - Open your web browser and go to `http://127.0.0.1:8000`

## Usage

1. **Upload Image**: Upload an image of the kitchen crockery items.
2. **Detection**: The system uses YOLOv5 to detect and identify the items.
3. **Billing**: Automatically generate a bill based on the detected items.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics.
- [Flask](https://flask.palletsprojects.com/) for web framework.

---

