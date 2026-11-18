import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,d_model,n_heads,block_size,d_rate):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.block_size = block_size
        self.dk = d_model//n_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(d_rate)
        self.register_buffer('mask',self.make_mask())


    def make_mask(self):
        # no device here: register_buffer moves it with model.to(device)
        return torch.triu(
        torch.full((self.block_size, self.block_size), float('-inf')),
        diagonal=1
    )

    def forward(self,x):
        b,t,c = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(b,t,self.n_heads,self.dk)
        k = k.view(b,t,self.n_heads,self.dk)
        v = v.view(b,t,self.n_heads,self.dk)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        score = q @ k.transpose(-2,-1) / (self.dk ** 0.5)

        score = score + self.mask[:t,:t]
        attn_weights = torch.softmax(score,dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v

        out = out.transpose(1,2).contiguous().view(b,t,self.d_model)

        return out


class Block(nn.Module):
    def __init__(self,d_model,n_heads,block_size,d_rate):
        super().__init__()
        self.attention = Attention(d_model,n_heads,block_size,d_rate)
        self.out_proj = nn.Linear(d_model,d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model,4*d_model),
                                 nn.GELU(),
                                 nn.Linear(4*d_model,d_model))
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(d_rate)

    def forward(self,x):
        out = self.attention(x)
        out = self.out_proj(out)
        x = x + self.dropout(self.layernorm1(out))
        x = x + self.dropout(self.layernorm2(self.ffn(x)))
        return x


class Model(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,n_layers,block_size,d_rate):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pos = nn.Embedding(block_size,d_model)
        self.layer = nn.ModuleList(Block(d_model,n_heads,block_size,d_rate) for _ in range(n_layers))
        self.layernorm3 = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        b,t = x.shape
        x = self.embed(x)
        pos = self.pos(torch.arange(t,device=x.device))
        x = x + pos

        for layer in self.layer:
            x = layer(x)
        x = self.layernorm3(x)
        return self.output(x)
