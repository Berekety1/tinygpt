from tokenizerv2 import return_data
import torch
import random
import config

data = return_data()

data = torch.tensor(data,dtype=torch.long,device=config.device)


def batching(batch_size,sequence_length,data):
    x = []
    y = []

    for i in range(batch_size):
        s = random.randint(0,len(data)-1-sequence_length)
        x.append(data[s:s+sequence_length])
        y.append(data[s+1:s+sequence_length+1])
    return torch.stack(x), torch.stack(y)
