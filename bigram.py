import torch
import torch.nn as nn
from batching import batching, return_train_data, return_data,d,e,decode


class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.W = nn.Embedding(vocab_size,vocab_size)

    def forward(self,x):
        return self.W(x)


class Bigram_Hybrid(nn.Module):
    def __init__(self,vocab_size = 66, d_model = 32, block_size = 16):
        super().__init__()
        self.token_embeeding = nn.Embedding(vocab_size,d_model)
        self.positional_encoding = nn.Embedding(block_size,d_model)
        self.model = nn.Sequential(nn.Linear(d_model,64),
                                   nn.ReLU(),
                                   nn.Linear(64,vocab_size))

    def forward(self,x):
            B,T = x.shape
            emb = self.token_embeeding(x)
            pos = self.positional_encoding(torch.arange(T, device=x.device))
            return self.model(emb+pos)












vocab_size = len(d)

bigram = Bigram(vocab_size)
bigram_hybrid = Bigram_Hybrid()
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.AdamW(
    bigram.parameters(),
    lr=0.001
)
optmizer2 = torch.optim.AdamW(bigram_hybrid.parameters(), lr=0.01)
n = 16
t = 8






steps = 20000


train_data = return_train_data()
def train_bigram(optimizer,model):
 for i in range(steps):
    optimizer.zero_grad()
    x,y = batching(n,t,train_data)
    logits = model(x)
    logits = logits.reshape(-1,vocab_size)
    target = y.reshape(-1)
    loss = criterion(logits,target)
    if i % 1000 == 0:
        print(f'step{i+1}: loss {loss.item()}')
    loss.backward()
    optimizer.step()


@torch.no_grad
def generate(size,starting_token):
    generated_words= [starting_token]
    for _ in range(size):
        logit = bigram(torch.tensor(starting_token,dtype=torch.long))
        prob = torch.softmax(logit,dim=-1)
        pred = torch.multinomial(prob, num_samples=1)
        starting_token = pred.item()
        generated_words.append(starting_token)
    return generated_words


@torch.no_grad
def generate(size,starting_token):
    generated_words= [starting_token]
    for _ in range(size):
        inp = torch.tensor(generated_words,dtype=torch.long)
     
        if len(inp) > 8:
            inp = inp[-8:]
        inp = inp.reshape(1,-1)
       
        logit = bigram_hybrid(inp)
  
        logit = logit.reshape(-1,vocab_size)
  
        prob = torch.softmax(logit,dim=-1)
        pred = torch.multinomial(prob, num_samples=1)
   
        generated_words.append(pred.flatten()[-1].item())

        

        
    return generated_words


train_bigram(optmizer2,bigram_hybrid)

gen = generate(300,30)


print(decode(gen,d))



