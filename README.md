# tinygpt

A small GPT-style language model built from scratch in PyTorch and trained on Tiny Shakespeare. It includes a byte-pair encoding (BPE) tokenizer written by hand, without libraries.

## What's inside

| File | What it does |
|---|---|
| `tokenizerv2.py` | BPE tokenizer: learns merges from the data, then encodes and decodes text |
| `tokenizer.py` | The simpler character-level tokenizer, kept for reference |
| `model.py` | Decoder-only transformer: causal multi-head self-attention, feed-forward layers, residuals, LayerNorm and dropout |
| `batching.py` | Samples random training windows from the encoded data |
| `train.py` | Training loop with a train/val split; saves the checkpoint with the best val loss |
| `generate.py` | Samples text from the model one token at a time |
| `config.py` | Every setting in one place |
| `main.py` | Entry point: loads a saved model if one exists, otherwise trains one, then generates text |

## Model

- 4 transformer blocks, 8 attention heads, `d_model` 64
- Context of 128 tokens
- BPE vocabulary of 512 tokens
- About 274k parameters

## Results

Best validation loss for each run. BPE losses are converted to loss per character so they can be compared with the character-level model.

| Tokenizer | Steps | Val loss per char |
|---|---|---|
| Character-level (65 tokens) | 5k | 1.657 |
| BPE, 512 tokens | 10k | **1.555** |
| BPE, 1024 tokens | 10k | 1.580 |

BPE beats character-level. A 1024-token vocabulary is slightly worse than 512: on a dataset this small, it leaves fewer tokens to train on, and the model overfits sooner.

## Run it

```bash
pip install torch
python main.py
```

The first run trains the tokenizer (about 1–2 minutes) and then the model. After that, `main.py` loads the saved model and goes straight to generating. Change any setting in `config.py` and it retrains automatically.

## Sample output

```
BUCKINGHAM:
But now I speak to me not, I throw me your mogures,
Than that would know mine lost shorter you. Yes.
You deserves welcome: hence would have insced!

CAPULET:
As shall of the troth of the profry!
```

## Data

The Tiny Shakespeare dataset (`data.txt`), from Andrej Karpathy's char-rnn project.
