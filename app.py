import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import gdown

# Use the correct Google Drive URL for downloading the model
model_url = 'https://drive.google.com/uc?id=1PO1ebNY67JhRE4LRf6hEnLykW3kDnMBF'

# Check if model already exists, otherwise download it
if not os.path.exists('leaf_disease_model.pth'):
    print("Downloading model...")
    gdown.download(model_url, 'leaf_disease_model.pth', quiet=False)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet101(weights=None)  # Change from pretrained=False to weights=None for newer versions
model.fc = torch.nn.Linear(model.fc.in_features, 6)  # Match the original number of classes (6)

# Load the saved model state_dict
state_dict = torch.load('leaf_disease_model.pth', map_location=device)

# Modify the state_dict to match the model architecture
new_state_dict = {}
for key, value in state_dict.items():
    if 'fc.0' in key:  # This means we have to rename 'fc.0' to 'fc'
        new_key = key.replace('fc.0', 'fc')
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Now load the modified state_dict into the model
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Define image transformations for prediction
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dynamically fetch class names from the dataset directory (train)
class_names = [
    'Apple___Black_rot',
    'Apple_Cedar_rust',
    'Apple_healthy',
    'Apple_scab',
    'Cherry_Powdery_mildew',
    'Cherry_healthy',
]

def predict_image(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is an image (optional but recommended)
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    # Create the uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    predicted_class = predict_image(img_path)
    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's port
    app.run(debug=True, host='0.0.0.0', port=port)  # Single app.run() is enough
