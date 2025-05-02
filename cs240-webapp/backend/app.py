import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import os
import sys

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Track detection status
detection_status = {"status": "pending"}

model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Fixed: changed pretrained to weights
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(image_path):
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)

    # Move image tensor to the same device as model (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_tensor = image_tensor.to(device)
    model.to(device)
    
    with torch.no_grad():
        try:
            # Perform the prediction
            prediction = model(image_tensor)
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)
    # Extract bounding boxes, labels, and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Only keep predictions with score > 0.5
    threshold = 0.5
    mask = scores > threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # Draw the bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a better font if available, fall back to default if not
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    # Generate unique filename based on original image name
    
    result_image_path = os.path.join(RESULTS_FOLDER, f"result_image.jpg")
    
    # Create color palette for different classes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get color from palette (cycling through colors)
        color = colors[i % len(colors)]
        
        # Convert box coordinates to integers to avoid drawing errors
        box_coords = [int(coord) for coord in box]
        
        # Draw bounding box
        draw.rectangle([tuple(box_coords[:2]), tuple(box_coords[2:])], outline=color, width=3)
        
        # Get the label text
        if 0 <= label < len(COCO_LABELS):  # Check if label index is valid
            label_text = f"{COCO_LABELS[label]}: {score:.2f}"
        else:
            label_text = f"Unknown ({label}): {score:.2f}"
        
        # Draw the label text with background for better visibility
        text_size = draw.textbbox((0, 0), label_text, font=font)[2:] 
        draw.rectangle([box_coords[0], box_coords[1] - text_size[1] - 4, 
                      box_coords[0] + text_size[0], box_coords[1]], 
                      fill=color)
        draw.text((box_coords[0], box_coords[1] - text_size[1] - 2), 
                label_text, fill="white", font=font)

    # Save the resulting image

    try:
        image.save(result_image_path)
        return result_image_path
    except Exception as e:
        print(f"Error saving result image: {e}")
        sys.exit(1)

def run_detection(filepath, result_filepath):
    global detection_status
    try:
        # subprocess.run(['python', 'detector.py', filepath, result_filepath])
        detect_objects(filepath)

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
    # Serve the result image from the 
    # 'results' directory
    detection_status["status"] = "pending"
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
