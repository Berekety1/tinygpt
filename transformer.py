import torch
import torch.nn as nn
import torch.nn.functional as F
from batching import batching, return_train_data, return_data,d,e,decode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class Head(nn.Module):
    def __init__(self,d_model, head_size, block_size, d_rate):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(d_model, head_size)
        self.query = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.register_buffer('mask',self.make_mask())
        self.dropout = nn.Dropout(d_rate)


    def make_mask(self):
        M = torch.zeros(self.block_size,self.block_size)
        mask = torch.triu(torch.ones_like(M,dtype=torch.bool), diagonal=1)
        result = M.masked_fill(mask,float('-inf'))
        return result



    def forward(self, X):
        B,T, C = X.shape
        key = self.key(X)
        query = self.query(X)
        value = self.value(X)
        score = query @ torch.transpose(key, 1, 2)
        score = score / self.head_size ** 0.5
        score = score + self.mask[:T, :T]
        score = F.softmax(score,dim=-1)
        score = self.dropout(score)
        score = score @ value
        return score



class Embed(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
        self.pos = nn.Embedding(block_size, d_model)

    def forward(self,x):
        B,T = x.shape
        emb = self.emb(x)
        pos = self.pos(torch.arange(T,device=x.device))
        return emb+pos





class MuliHeadAttention(nn.Module):
    def __init__(self,d_model, n_heads,block_size, vocab_size,d_rate):
        super().__init__()
        self.head_size = d_model//n_heads
        self.heads = nn.ModuleList(Head(d_model,self.head_size,block_size, d_rate) for _ in range(n_heads))
       


    def forward(self,X):
        outputs = []
        for head in self.heads:
            output = head(X)
            outputs.append(output)
        concat = torch.cat(outputs,dim=-1)
        return concat
       


class FinalHead(nn.Module):
    def __init__(self,vocab_size,n_heads,d_model,block_size,d_rate):
        super().__init__()
        
        self.multi = MuliHeadAttention(d_model,n_heads,block_size,vocab_size,d_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model,256),
                                 nn.GELU(),
                                 nn.Linear(256,d_model))
        self.out_proj = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(d_rate)

    def forward(self,X):
        out = self.multi(X)
        out = self.out_proj(out)
        
        
        X = X + self.dropout(self.layernorm1(out))
        X = X + self.dropout(self.layernorm2(self.ffn(X)))
        return X


class Model(nn.Module):
    def __init__(self,vocab_size,n_heads,d_model,block_size,n_layers,d_rate):
        super().__init__()
        self.embed = Embed(vocab_size,d_model,block_size)
        self.model = nn.ModuleList(FinalHead(vocab_size,n_heads,d_model,block_size,d_rate) for _ in range(n_layers))
        self.output = nn.Linear(d_model,vocab_size)
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self,X):
        X = self.embed(X)
        for layer in self.model:
            X = layer(X)
        X = self.layernorm(X)
        return self.output(X)
            


vocab_size = len(d)
d_model = 64
block_size = 128
n_heads = 8
n_layers = 4
d_rate = 0.2
train_percentage = 0.9
data = return_train_data()
train_data = data[:int(len(data)*train_percentage)]
validate_data = data[int(len(data)*train_percentage):]
#att = Attention(vocab_size,d_model,block_size,head_size)
#multi = MuliHeadAttention(d_model,8,block_size,vocab_size)
#embed = Embed(vocab_size,d_model,block_size)
#model = FinalHead(vocab_size,n_heads,d_model,block_size)
model = Model(vocab_size,n_heads,d_model,block_size,n_layers,d_rate)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

steps = 2500

def train():
    for i in range(steps):
        optimizer.zero_grad()
        x,y = batching(block_size,block_size,train_data)
        x = x.to(device)
        y = y.to(device)
        model.train() 
        logit = model(x)
        logit = logit.reshape(-1, vocab_size)
        target = y.reshape(-1)
        loss = criterion(logit,target)
        if i%500 == 0:
            print(f'Train loss at step {i} - {estimate_loss(train_data,10)}')
            print(f'Validate loss at step {i} - {estimate_loss(validate_data,10)}')
        loss.backward()
        optimizer.step()


@torch.no_grad

def estimate_loss(data,n):
    loss = 0
    for _ in range(n):
     x,y = batching(block_size,block_size,data)
     x = x.to(device)
     y = y.to(device)
     model.eval()
     logit = model(x)
     logit = logit.reshape(-1,vocab_size)
     target = y.reshape(-1)
     l = criterion(logit,target).item()
     loss = loss + l

    return loss/n





@torch.no_grad
def generate(size,starting_token):
    generated_words= [starting_token]
    for _ in range(size):
        
        inp = torch.tensor(generated_words,dtype=torch.long,device=device)
     
        if len(inp) > block_size:
            inp = inp[-block_size:]
        inp = inp.reshape(1,-1)
        model.eval()
        logit = model(inp)
  
        logit = logit.reshape(-1,vocab_size)
  
        prob = torch.softmax(logit,dim=-1)
        pred = torch.multinomial(prob, num_samples=1)
   
        generated_words.append(pred.flatten()[-1].item())
    return generated_words


train()
gen = generate(300,e['ROMEO'])


print(decode(gen,d))

