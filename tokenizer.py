import os

def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data.txt")
    with open(path,'r') as f:
        data = f.read()
    return data

data = load_data()
chars = sorted(set(data))
stoi = {chars[i]:i for i in range(len(chars))}
itos = {i:chars[i] for i in range(len(chars))}


def encode(data):
    tokens = []
    for char in data:
        tokens.append(stoi[char])
    return tokens

def decode(data):
    string = []
    for num in data:
        string.append(itos[num])
    return "".join(string)

def return_data():
    return encode(data)
    






