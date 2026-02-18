import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from config import validate_config
from data_processor import batch_commute_to_features, extract_predictions
from model import (
    train_model, load_checkpoint, save_checkpoint, list_checkpoints,
    generate_optimal
)

app = Flask(__name__)
CORS(app)

DATA_FILE = "commute_data.json"
MODELS_DIR = "models"

# Global model state
model_state = {
    "model": None,
    "config": None,
    "training_info": None,
    "last_checkpoint": None
}

def load_data():
    """Load existing data or return empty list"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    """Save data to JSON file"""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/api/commute", methods=["POST"])
def save_commute():
    """Save a new commute entry"""
    entry = request.json
    entry["timestamp"] = datetime.now().isoformat()

    data = load_data()
    data.append(entry)
    save_data(data)

    return jsonify({"success": True, "total": len(data)}), 201

@app.route("/api/commutes", methods=["GET"])
def get_commutes():
    """Get all commute entries"""
    data = load_data()
    return jsonify({"count": len(data), "commutes": data})

@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Export data as CSV"""
    import csv
    import io

    data = load_data()
    if not data:
        return {"error": "No data"}, 404

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

    return output.getvalue(), 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": 'attachment; filename="commute_data.csv"'
    }

@app.route("/api/health", methods=["GET"])
def get_health():
    """Return health status"""
    data = load_data()
    return jsonify({"status": "healthy"})


@app.route("/api/train", methods=["POST"])
def train():
    """
    Train the model on commute data with configurable parameters.

    Request body:
    {
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "train_val_split": 0.8,
        "loss_weights": {
            "features": 1.0,
            "travel_time": 1.0,
            "pleasure": 0.5
        }
    }
    """
    try:
        # Load data
        data = load_data()
        if len(data) < 2:
            return jsonify({"error": "Need at least 2 commute samples to train"}), 400

        # Validate and merge config
        user_config = request.json or {}
        config = validate_config(user_config)

        # Convert commute data to features
        X, mask = batch_commute_to_features(data)

        # Extract or calculate labels from data
        y_travel_time = []
        y_pleasure = []

        for d in data:
            # Calculate travel time from departure/arrival times
            if "departureTime" in d and "arrivalTime" in d:
                from data_processor import time_str_to_minutes
                try:
                    departure_min = time_str_to_minutes(d["departureTime"])
                    arrival_min = time_str_to_minutes(d["arrivalTime"])
                    travel_time = max(0, arrival_min - departure_min)
                except:
                    travel_time = 30.0
            else:
                travel_time = d.get("travelTime", 30.0)

            # Get pleasure rating (normalize from 1-10 to 0-1)
            pleasure_rating = d.get("pleasureRating", 5)
            if isinstance(pleasure_rating, str):
                pleasure_rating = float(pleasure_rating)
            pleasure = max(0, min(1, pleasure_rating / 10.0))

            y_travel_time.append([float(travel_time)])
            y_pleasure.append([float(pleasure)])

        y_travel_time = np.array(y_travel_time, dtype=np.float32)
        y_pleasure = np.array(y_pleasure, dtype=np.float32)

        # Train model
        model, training_info = train_model(X, mask, y_travel_time, y_pleasure, config)

        # Save checkpoint
        checkpoint_path = save_checkpoint(model, config, MODELS_DIR)

        # Update global state
        model_state["model"] = model
        model_state["config"] = config
        model_state["training_info"] = training_info
        model_state["last_checkpoint"] = checkpoint_path

        return jsonify({
            "status": "training_complete",
            "samples_used": len(data),
            "epochs_trained": config.get("epochs"),
            "final_loss": training_info.get("final_loss"),
            "final_val_loss": training_info.get("final_val_loss"),
            "model_saved_at": checkpoint_path
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Generate optimized commute recommendation.

    Request body:
    {
        "day": "Monday",
        "startTime": 540,
        "known_features": {
            "route": 1,
            "stops": [0, 1]
        },
        "inference_params": {
            "n_samples": 50,
            "time_weight": 1.0,
            "pleasure_weight": 1.0
        }
    }
    """
    try:
        if model_state["model"] is None:
            return jsonify({"error": "Model not trained yet. Call /api/train first."}), 400

        request_data = request.json or {}

        # Build feature vector from request
        commute_query = {
            "day": request_data.get("day", "Monday"),
            "startTime": request_data.get("startTime", 540),
        }

        # Add known features
        known_features = request_data.get("known_features", {})
        if "route" in known_features:
            commute_query["route"] = known_features["route"]
        if "stops" in known_features:
            commute_query["stops"] = known_features["stops"]
        else:
            commute_query["stops"] = []

        # Convert to features
        from data_processor import commute_to_features
        x, mask = commute_to_features(commute_query)
        x = x.reshape(1, -1)
        mask = mask.reshape(1, -1)

        # Merge inference params
        inference_params = request_data.get("inference_params", {})
        config = model_state["config"].copy()
        config["inference_params"].update(inference_params)

        # Generate prediction
        best_score, best_features, best_travel_time, best_pleasure = generate_optimal(
            model_state["model"],
            x,
            mask,
            config
        )

        # Convert back to human-readable format
        recommendation = extract_predictions(best_features)

        return jsonify({
            "best_score": float(best_score),
            "recommendation": recommendation,
            "predicted_travel_time": float(best_travel_time),
            "predicted_pleasure": float(best_pleasure)
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/model/status", methods=["GET"])
def model_status():
    """Get status of the trained model."""
    if model_state["model"] is None:
        return jsonify({
            "trained": False,
            "message": "No model trained yet"
        }), 200

    return jsonify({
        "trained": True,
        "last_checkpoint": model_state.get("last_checkpoint"),
        "config": model_state.get("config"),
        "training_info": model_state.get("training_info")
    }), 200


@app.route("/api/model/checkpoints", methods=["GET"])
def list_model_checkpoints():
    """List all available model checkpoints."""
    checkpoints = list_checkpoints(MODELS_DIR)
    return jsonify({
        "total": len(checkpoints),
        "checkpoints": checkpoints
    }), 200


@app.route("/api/model/load", methods=["POST"])
def load_model():
    """
    Load a specific model checkpoint.

    Request body:
    {
        "model_path": "/absolute/path/to/checkpoint.h5"
    }
    """
    try:
        request_data = request.json or {}
        model_path = request_data.get("model_path")

        if not model_path:
            return jsonify({"error": "model_path required"}), 400

        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404

        # Load model
        model = load_checkpoint(model_path)

        # Load config if available
        timestamp = os.path.basename(model_path).replace("checkpoint_", "").replace(".h5", "")
        config_path = os.path.join(MODELS_DIR, f"config_{timestamp}.json")

        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        # Update state
        model_state["model"] = model
        model_state["config"] = config
        model_state["last_checkpoint"] = model_path

        return jsonify({
            "status": "model_loaded",
            "model_path": model_path,
            "config": config
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


@app.route("/api/model/save", methods=["POST"])
def save_model():
    """Manually save the current model."""
    try:
        if model_state["model"] is None:
            return jsonify({"error": "No model to save"}), 400

        checkpoint_path = save_checkpoint(model_state["model"], model_state["config"], MODELS_DIR)
        model_state["last_checkpoint"] = checkpoint_path

        return jsonify({
            "status": "model_saved",
            "model_path": checkpoint_path
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"✓ API running at http://localhost:5000")
    print(f"Data file: {os.path.abspath(DATA_FILE)}")
    app.run(debug=True, port=5000)
