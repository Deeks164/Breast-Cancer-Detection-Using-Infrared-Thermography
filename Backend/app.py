import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# SAME MODEL BUT WITH 3-CLASS OUTPUT
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # 👈 3 OUTPUTS
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model_path = os.path.join('model', 'cnn_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(1).item()

    classes = ["Invalid Image – Not a Breast Thermography Scan", "Normal", "Sick"]
    prediction = classes[pred_class]

    return jsonify({
        'prediction': prediction,
        'confidence': float(torch.softmax(output, dim=1)[0][pred_class])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
