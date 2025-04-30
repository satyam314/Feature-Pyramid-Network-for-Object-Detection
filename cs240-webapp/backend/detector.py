import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os
import sys

# Ensure the results directory exists
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the pretrained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to perform detection
def detect_objects(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)

    # Move image tensor to the same device as model (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_tensor = image_tensor.to(device)
    model.to(device)
    
    with torch.no_grad():
        # Perform the prediction
        prediction = model(image_tensor)

    # Extract bounding boxes and labels
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Only keep predictions with score > 0.5
    threshold = 0.5
    boxes = boxes[scores > threshold]
    labels = labels[scores > threshold]
    
    # Draw the bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle([tuple(box[:2]), tuple(box[2:])], outline="red", width=3)

    # Save the resulting image
    result_image_path = os.path.join(RESULTS_DIR, "result_image.jpg")
    image.save(result_image_path)

    return result_image_path

# Main function to be called
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result_image = detect_objects(image_path)
    print(f"Result image saved at: {result_image}")
