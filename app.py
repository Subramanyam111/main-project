import os
import logging
from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.cnn_model import CNN
import json  # Import the json module

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "models/cnn_weed_classifier.pth"
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file '{MODEL_PATH}' not found. Please train or provide a valid model.")
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

logging.info("Loading trained model...")
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
logging.info("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class_labels = ["Carpetweeds_1", "Crabgrass_2", "Eclipta_3", "Goosegrass_4", "Morningglory_5", "Nutsedge_6", "PalmerAmaranth_7", "Prickly Sida_8", "Purslane_9", "Ragweed_10", "Sicklepod_11", "SpottedSpurge_12", "SpurredAnoda_13", "Swinecress_14", "Waterhemp_15"]

# Load weed data from JSON file
try:
    with open('weed_data.json', 'r') as f:
        weed_data = json.load(f)
except FileNotFoundError:
    logging.error("weed_data.json not found.")
    weed_data = {}  # Use an empty dictionary as a fallback
except json.JSONDecodeError:
    logging.error("Error decoding weed_data.json.")
    weed_data = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.warning("No file uploaded in request.")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        logging.warning("No file selected by user.")
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    logging.info(f"Image saved to {filepath}")

    try:
        image = Image.open(filepath).convert('RGB')
        image = transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    with torch.inference_mode():
        output = model(image)
        logging.info(f"Output Shape: {output.shape}")
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

        if predicted.item() >= len(class_labels):
            logging.error(f"Invalid prediction index {predicted.item()} (out of range).")
            return jsonify({"error": f"Prediction index {predicted.item()} out of range."}), 500

        label = class_labels[predicted.item()]
        weed_info = weed_data.get(label, {})

    logging.info(f"Prediction: {label}, Confidence: {confidence:.4f}")

    return render_template('result.html', image=filepath, label=label, confidence=round(confidence, 4), weed_info=weed_info)

if __name__ == '__main__':
    app.run(debug=True)