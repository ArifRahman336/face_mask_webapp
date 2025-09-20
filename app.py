from flask import Flask, render_template, Response, request, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# App setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Models
detection_model = YOLO("bt.pt")
classification_model = load_model("mobilenetv25.h5")

# Webcam setup
camera = cv2.VideoCapture(0)

# ---------- Realtime Video Feed Generators ----------
def generate_frames_detection():
    while True:
        success, frame = camera.read()
        if not success:
            break
        results = detection_model(frame)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_classification():
    labels = ['with_mask', 'without_mask']
    while True:
        success, frame = camera.read()
        if not success:
            break
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = classification_model.predict(img)
        class_idx = np.argmax(prediction[0])
        class_label = labels[class_idx]
        cv2.putText(frame, f"Prediction: {class_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------- ROUTES ----------

@app.route('/')
def index():
    return render_template('index.html')  # Step 1: choose static or realtime

@app.route('/select_source')
def select_source():
    source = request.args.get('source')
    return render_template('select_model.html', source=source)  # Step 2

# ---------- Real-time Routes ----------

@app.route('/start_realtime_detection')
def start_realtime_detection():
    return render_template('result.html', model_type='detection')

@app.route('/start_realtime_classification')
def start_realtime_classification():
    return render_template('result.html', model_type='classification')

@app.route('/video_feed/<model_type>')
def video_feed(model_type):
    if model_type == 'detection':
        return Response(generate_frames_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif model_type == 'classification':
        return Response(generate_frames_classification(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid model type", 400

# ---------- Static Image Routes ----------

@app.route('/start_static_<model_type>')
def start_static_model(model_type):
    if model_type not in ['classification', 'detection']:
        return "Invalid model type", 400
    return render_template('upload_image.html', model_type=model_type)

@app.route('/process_static_<model_type>', methods=['POST'])
def process_static_model(model_type):
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if model_type == 'classification':
        # Classification logic
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        labels = ['with_mask', 'without_mask']
        prediction = classification_model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        class_label = labels[class_idx]
        result_text = f"Prediction: {class_label}"
        return render_template('display_result.html', image_file=filename, result=result_text)

    elif model_type == 'detection':
        # Detection logic
        img = cv2.imread(filepath)
        results = detection_model(img)
        annotated_img = results[0].plot()
        result_filename = 'result_' + filename
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_img)
        return render_template('display_result.html', image_file=result_filename, result="Detection Complete")

    else:
        return "Invalid model type", 400

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
