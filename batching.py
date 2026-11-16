from tokenizer import read_data, tokenize, encode,decode,make_data
import random
import torch
data = read_data()
d,e = tokenize(sorted(set(data)))

train_data = make_data(data,e)
print(len(train_data))

def batching(n,t,d:torch.Tensor):
    x = []
    y = []
    for i in range(n):
        s = random.randint(0,len(d)-t-1)
        
        x.append(d[s:s+t])
        y.append(d[s+1:s+t+1])

    return torch.stack(x),torch.stack(y)

def return_train_data():
    return train_data

def return_data():
    return sorted(set(data))



