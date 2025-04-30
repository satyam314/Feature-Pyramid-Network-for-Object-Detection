import os
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Track detection status
detection_status = {"status": "pending"}

def run_detection(filepath, result_filepath):
    global detection_status
    try:
        subprocess.run(['python', 'detector.py', filepath, result_filepath])

        # After detection, update status
        detection_status["status"] = "done"
        detection_status["result_image"] = os.path.basename(result_filepath)
    except Exception as e:
        detection_status["status"] = "error"
        detection_status["error"] = str(e)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    global detection_status
    detection_status = {"status": "pending"}  # Reset status on each new upload

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded file to disk
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run the detector script (synchronously this time)
    threading.Thread(target=run_detection, args=(filepath, 'result_image.jpg')).start()
   
    return jsonify({'message': 'File uploaded and detection started.'}), 200

@app.route('/api/status', methods=['GET'])
def status():
    global detection_status
    return jsonify(detection_status)

@app.route('/results/<filename>', methods=['GET'])
def serve_result_image(filename):
    # Serve the result image from the 'results' directory
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
