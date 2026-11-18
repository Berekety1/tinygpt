import os
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- model (same as v1) ----
vocab_size = 512  # 65 for Tiny Shakespeare
d_model = 64
n_heads = 8
n_layers = 4
block_size = 128          # context length: how many tokens the model sees at once
d_rate = 0.2              # dropout

# ---- training ----
batch_size = 128
lr = 0.001
steps = 10000
eval_interval = 500       # check train/val loss every this many steps
eval_batches = 10         # how many batches estimate_loss averages over
train_percentage = 0.9

# ---- files ----
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinygpt_best.pth")

# Everything needed to rebuild the model. Saved inside the checkpoint so
# main.py can tell whether a saved model matches the current settings.
MODEL_CONFIG = dict(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size,
    d_rate=d_rate,
)
