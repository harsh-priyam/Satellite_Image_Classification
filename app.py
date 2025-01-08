from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
app = Flask(__name__)

# Load the model and weights
model = models.resnet18()  # Do not use pretrained=True
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 4)
# Explicitly map the model to CPU
model.load_state_dict(torch.load("resnet_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['cloudy', 'desert', 'green_area', 'water']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected!"}), 400

    try:
        # Open the image
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
        
        result = {
            "class": class_names[predicted_class],
            "confidence": f"{probabilities[predicted_class].item():.4f}"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

PORT = 8000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
