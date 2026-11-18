import torch
import torch.nn as nn
import config
from batching import batching,data

split = int(len(data)*config.train_percentage)
train_data = data[:split]
val_data = data[split:]

criterion = nn.CrossEntropyLoss()


def save_checkpoint(model,step,val_loss):
    torch.save({
        "config": config.MODEL_CONFIG,
        "model": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, config.checkpoint_path)


@torch.no_grad()
def estimate_loss(model,data,n):
    model.eval()
    loss = 0
    for i in range(n):
        x,y = batching(config.batch_size,config.block_size,data)
        logit = model(x)
        logit = logit.reshape(-1,config.vocab_size)
        target = y.reshape(-1)
        loss += criterion(logit,target).item()
    model.train()
    return loss/n


def train(model):
    optim = torch.optim.AdamW(model.parameters(),lr=config.lr)
    best_val = float('inf')
    model.train()

    for i in range(config.steps):
        if i % config.eval_interval == 0 or i == config.steps - 1:
            train_loss = estimate_loss(model,train_data,config.eval_batches)
            val_loss = estimate_loss(model,val_data,config.eval_batches)
            msg = f"STEP {i} | TRAIN LOSS {train_loss:.4f} | VAL LOSS {val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model,i,val_loss)
                msg += "  <- best so far, saved"
            print(msg)

        optim.zero_grad()
        x,y = batching(config.batch_size,config.block_size,train_data)
        logit = model(x)
        logit = logit.reshape(-1,config.vocab_size)
        target = y.reshape(-1)
        loss = criterion(logit,target)
        loss.backward()
        optim.step()

    return best_val
