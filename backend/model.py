"""ML model architecture and training logic."""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from datetime import datetime
from typing import Tuple, List, Dict, Any
from config import TOTAL_FEATURES


@tf.keras.utils.register_keras_serializable(package="commutement")
class MaskedFeatureLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        features_true, features_pred, mask = inputs
        unknown_mask = 1.0 - mask
        diff = (features_true - features_pred) * unknown_mask
        squared = diff * diff
        summed = tf.reduce_sum(squared, axis=-1)
        masked_count = tf.reduce_sum(unknown_mask, axis=-1) + 1e-6
        mse_per_sample = summed / masked_count
        self.add_loss(tf.reduce_mean(mse_per_sample))
        return features_pred


def build_encoder(input_dim: int, latent_dim: int = 8) -> Model:
    """
    Build encoder model.

    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension

    Returns:
        Keras Model for encoder
    """
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    z = layers.Dense(latent_dim)(x)
    return Model(inputs, z, name='encoder')


def build_decoder(latent_dim: int, output_dim: int) -> Model:
    """
    Build decoder model with three output heads.

    Args:
        latent_dim: Latent space dimension
        output_dim: Output feature dimension

    Returns:
        Keras Model for decoder
    """
    z = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(z)
    x = layers.Dense(64, activation='relu')(x)

    features_out = layers.Dense(output_dim, name='features_out')(x)
    travel_time_out = layers.Dense(1, name='travel_time_out')(x)
    pleasure_out = layers.Dense(1, name='pleasure_out')(x)

    return Model(z, [features_out, travel_time_out, pleasure_out], name='decoder')


def build_model(config: Dict[str, Any]) -> Model:
    """
    Build complete masked conditional autoencoder.

    Args:
        config: Configuration dictionary with:
            - latent_dim: Latent dimension
            - loss_weights: Dict with 'features', 'travel_time', 'pleasure' weights

    Returns:
        Compiled Keras Model
    """
    latent_dim = config.get("latent_dim", 8)
    loss_weights = config.get("loss_weights", {})

    feature_input = layers.Input(shape=(TOTAL_FEATURES,), name='features')
    mask_input = layers.Input(shape=(TOTAL_FEATURES,), name='mask')

    masked_features = layers.Multiply()([feature_input, mask_input])
    model_input = layers.Concatenate()([masked_features, mask_input])

    encoder = build_encoder(model_input.shape[-1], latent_dim)
    decoder = build_decoder(latent_dim, TOTAL_FEATURES)

    z = encoder(model_input)
    decoder_outputs = decoder(z)
    features_hat = decoder_outputs[0]
    travel_time_hat = decoder_outputs[1]
    pleasure_hat = decoder_outputs[2]

    features_hat = MaskedFeatureLossLayer(name='features_out')([feature_input, features_hat, mask_input])
    travel_time_hat = layers.Identity(name='travel_time_out')(travel_time_hat)
    pleasure_hat = layers.Identity(name='pleasure_out')(pleasure_hat)

    model = Model(
        inputs=[feature_input, mask_input],
        outputs=[features_hat, travel_time_hat, pleasure_hat],
        name='autoencoder'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get("learning_rate", 1e-3)),
        loss={
            "features_out": "mse",
            "travel_time_out": "mse",
            "pleasure_out": "mse"
        },
        loss_weights={
            "features_out": 0.0,
            "travel_time_out": loss_weights.get("travel_time", 1.0),
            "pleasure_out": loss_weights.get("pleasure", 0.5)
        }
    )

    return model


def train_model(
    X: np.ndarray,
    mask: np.ndarray,
    y_travel_time: np.ndarray,
    y_pleasure: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[Model, Dict[str, Any]]:
    """
    Train the model on commute data.

    Args:
        X: Features matrix of shape (n_samples, TOTAL_FEATURES)
        mask: Mask matrix of shape (n_samples, TOTAL_FEATURES)
        y_travel_time: Travel time labels of shape (n_samples, 1)
        y_pleasure: Pleasure labels of shape (n_samples, 1)
        config: Configuration dictionary

    Returns:
        Tuple of (trained_model, training_info_dict)
    """
    model = build_model(config)

    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 32)

    history = model.fit(
        [X, mask],
        {
            "features_out": X,
            "travel_time_out": y_travel_time,
            "pleasure_out": y_pleasure
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_split=1 - config.get("train_val_split", 0.8),
        verbose=0
    )

    return model, {
        "final_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "epochs_trained": epochs,
        "samples": len(X)
    }


def generate_optimal(
    model: Model,
    x: np.ndarray,
    mask: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate optimal commute recommendation via candidate sampling.

    Args:
        model: Trained autoencoder model
        x: Query features of shape (1, TOTAL_FEATURES)
        mask: Query mask of shape (1, TOTAL_FEATURES)
        config: Configuration with inference_params

    Returns:
        Tuple of (best_score, best_features, best_travel_time, best_pleasure)
    """
    inference_params = config.get("inference_params", {})
    n_samples = inference_params.get("n_samples", 50)
    time_weight = inference_params.get("time_weight", 1.0)
    pleasure_weight = inference_params.get("pleasure_weight", 1.0)

    candidates = []

    for _ in range(n_samples):
        f_hat, t_hat, p_hat = model.predict([x, mask], verbose=0)

        # Force known query features to remain fixed
        f_hat = np.where(mask == 1, x, f_hat)

        score = (
            -time_weight * t_hat[0, 0] +
            pleasure_weight * p_hat[0, 0]
        )

        candidates.append({
            "score": float(score),
            "features": f_hat[0],
            "travel_time": float(t_hat[0, 0]),
            "pleasure": float(p_hat[0, 0])
        })

    best = max(candidates, key=lambda c: c["score"])
    return best["score"], best["features"], best["travel_time"], best["pleasure"]


def save_checkpoint(model: Model, config: Dict[str, Any], models_dir: str = "models") -> str:
    """
    Save model checkpoint to disk.

    Args:
        model: Trained model
        config: Configuration used for training
        models_dir: Directory to save models

    Returns:
        Path to saved model
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"checkpoint_{timestamp}.h5")
    config_path = os.path.join(models_dir, f"config_{timestamp}.json")

    model.save(model_path)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return model_path


def load_checkpoint(model_path: str) -> Model:
    """
    Load model checkpoint from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"MaskedFeatureLossLayer": MaskedFeatureLossLayer},
        compile=False
    )


def list_checkpoints(models_dir: str = "models") -> List[Dict[str, str]]:
    """
    List all saved checkpoints.

    Args:
        models_dir: Directory containing models

    Returns:
        List of dicts with 'model_path' and 'config_path'
    """
    if not os.path.exists(models_dir):
        return []

    checkpoints = []
    for filename in sorted(os.listdir(models_dir), reverse=True):
        if filename.startswith("checkpoint_") and filename.endswith(".h5"):
            timestamp = filename.replace("checkpoint_", "").replace(".h5", "")
            model_path = os.path.join(models_dir, filename)
            config_path = os.path.join(models_dir, f"config_{timestamp}.json")

            checkpoint_info = {
                "timestamp": timestamp,
                "model_path": os.path.abspath(model_path),
                "config_path": os.path.abspath(config_path) if os.path.exists(config_path) else None,
                "model_exists": os.path.exists(model_path),
                "config_exists": os.path.exists(config_path)
            }
            checkpoints.append(checkpoint_info)

    return checkpoints
