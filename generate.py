import torch
import config


@torch.no_grad()
def generate(model,starting_tokens,n):
    model.eval()
    generated = list(starting_tokens)   # copy, so the caller's list isn't changed

    for _ in range(n):
        x = torch.tensor(generated[-config.block_size:],dtype=torch.long,device=config.device)
        x = x.reshape(1,-1)
        logit = model(x)
        logit = logit[0,-1,:]           # only the last position predicts the next token
        prob = torch.softmax(logit,dim=-1)
        pred = torch.multinomial(prob,num_samples=1)
        generated.append(pred.item())
    return generated
