import torch
import torch.nn as nn
import torch.nn.functional as F
from batching import batching, return_train_data, return_data,d,e,decode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class Head(nn.Module):
    def __init__(self,d_model, head_size, block_size, vocab_size):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(d_model, head_size)
        self.query = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.mask = self.make_mask()


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
        pos = self.pos(torch.arange(T))
        return emb+pos



class Attention(nn.Module):
    def __init__(self, vocab_size, d_model,block_size, head_size):
        super().__init__()
        self.head = Head(d_model,head_size,block_size,vocab_size)
        self.embed = Embed(vocab_size,d_model, block_size)


    def forward(self, X):
        emb = self.embed(X)
        output = self.head(emb)
        return output


class MuliHeadAttention(nn.Module):
    def __init__(self,d_model, n_heads,block_size, vocab_size):
        super().__init__()
        self.head_size = d_model//n_heads
        self.heads = nn.ModuleList(Head(d_model,self.head_size,block_size, vocab_size) for _ in range(n_heads))
        self.output_proj = nn.Linear(d_model,d_model)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self,X):
        outputs = []
        for head in self.heads:
            output = head(X)
            outputs.append(output)
        concat = torch.cat(outputs,dim=-1)
        output = self.output_proj(concat)
        return self.layer_norm(X + output)





train_data = return_train_data()
vocab_size = len(d)
d_model = 64
block_size = 128
head_size = 32


att = Attention(vocab_size,d_model,block_size,head_size)
multi = MuliHeadAttention(d_model,8,block_size,vocab_size)
embed = Embed(vocab_size,d_model,block_size)




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(att.parameters(), lr=0.001)

steps = 10

def train():
    for i in range(steps):
        optimizer.zero_grad()
        x,y = batching(block_size,block_size,train_data)
        x = embed(x)
        logit = multi(x)
        logit = logit.reshape(-1, vocab_size)
        target = y.reshape(-1)
        loss = criterion(logit,target)
        if i%500 == 0:
            print(f'step{i} Loss {loss.item()}')
        loss.backward()
        optimizer.step()


@torch.no_grad
def generate(size,starting_token):
    generated_words= [starting_token]
    for _ in range(size):
        
        inp = torch.tensor(generated_words,dtype=torch.long)
     
        if len(inp) > block_size:
            inp = inp[-block_size:]
        inp = inp.reshape(1,-1)
        inp = embed(inp)
        logit = multi(inp)
  
        logit = logit.reshape(-1,vocab_size)
  
        prob = torch.softmax(logit,dim=-1)
        pred = torch.multinomial(prob, num_samples=1)
   
        generated_words.append(pred.flatten()[-1].item())
    return generated_words


train()
gen = generate(300,30)


print(decode(gen,d))

