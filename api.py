from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Define your model
class FraudModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Load model
model = FraudModel()
model.load_state_dict(torch.load("global_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route("/")
def home():
    return "Credit Card Fraud Detection API (via Federated Learning)"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != 30:
            return jsonify({"error": "Exactly 30 features are required."}), 400

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            confidence = output.item()
            prediction = int(confidence > 0.5)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
