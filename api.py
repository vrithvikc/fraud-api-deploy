from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define your model
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
        # Receive JSON with 'features' key containing 29 values
        data = request.get_json(force=True)
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)

        # Predict
        input_tensor = torch.tensor(features)
        with torch.no_grad():
            output = model(input_tensor).item()
        prediction = int(output > 0.5)

        return jsonify({
            "prediction": prediction,
            "confidence": round(output, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
