import torch


def read_data():
    with open("data.txt", 'r') as f:
        data = f.read()
    return data

def tokenize(data):
    decode_dict = {}
    encode_dict = {}
    for i in range(len(data)):
        encode_dict[data[i]] = i+1
        decode_dict[i+1] = data[i]

    decode_dict[0] = "UNK"
    return decode_dict,encode_dict

def encode(text,encode_dict):
    m = []
    for char in text:
        if char in encode_dict:
            m.append(encode_dict[char])
        else:
            m.append(0)
    return m
def decode(m,decode_dict):
    l = []
    for w in m:
        if w in decode_dict:
            l.append(decode_dict[w])

    return "".join(l)

def make_data(data,e):
    return torch.tensor(encode(data,e), dtype=torch.long)




