import os
import torch
import config
from model import Model
from tokenizerv2 import encode,decode
from train import train
from generate import generate


def load_saved_model():
    """Return the saved model if one exists and matches config.py, otherwise None."""
    if not os.path.exists(config.checkpoint_path):
        print("No saved model found.")
        return None

    checkpoint = torch.load(config.checkpoint_path,map_location=config.device)
    if checkpoint["config"] != config.MODEL_CONFIG:
        print("Saved model was trained with different settings than config.py.")
        return None

    model = Model(**checkpoint["config"]).to(config.device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded saved model (step {checkpoint['step']}, val loss {checkpoint['val_loss']:.4f}).")
    return model


def main():
    model = load_saved_model()

    if model is None:
        print("Training a new model...")
        model = Model(**config.MODEL_CONFIG).to(config.device)
        train(model)
        model = load_saved_model()   # reload the best checkpoint, not the last step

    prompt = "ROMEO"
    output = generate(model,encode(prompt),700)
    print(decode(output))


if __name__ == "__main__":
    main()
