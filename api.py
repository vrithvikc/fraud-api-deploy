from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn

# Define the model
class FraudModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(29, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Load model
model = FraudModel()
model.load_state_dict(torch.load("global_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route("/")
def home():
    return "✅ Credit Card Fraud Detection API is live."

# Handle preflight requests manually
@app.route("/predict", methods=["OPTIONS"])
def handle_options():
    return '', 204

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key in request body"}), 400
        
        features = data["features"]

        if not isinstance(features, list) or len(features) != 30:
            return jsonify({"error": "Invalid input. 30 numerical features expected."}), 400

        input_tensor = torch.tensor(features).float().unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            confidence = output.item()
            prediction = 1 if confidence > 0.5 else 0

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
