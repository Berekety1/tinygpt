import os
import json
from config import vocab_size
HERE = os.path.dirname(os.path.abspath(__file__))
MERGES_PATH = os.path.join(HERE,"merges.json")     # the tokenizer itself
ENCODED_PATH = os.path.join(HERE,"encoded.json")   # data.txt already encoded (cache)




def load_data():
    path = os.path.join(HERE,"data.txt")
    with open(path,'r') as f:
        data = f.read()
    return data

data = load_data()
chars = sorted(set(data))
stoi = {chars[i]:i for i in range(len(chars))}
itos = {i:chars[i] for i in range(len(chars))}


def encode_chars(text):
    tokens = []
    for char in text:
        tokens.append(stoi[char])
    return tokens


def findadjacent(data_list):
    t = {}
    for i in range(len(data_list)-1):
        if (data_list[i],data_list[i+1]) in t.keys():
            t[(data_list[i],data_list[i+1])] += 1
        else:
            t[(data_list[i],data_list[i+1])] = 1
    t = max(t,key=t.get)
    return t


def merge(data_list, pair, new_id):
    data = []
    i = 0
    while i < len(data_list):
       if i+1 < len(data_list) and (data_list[i],data_list[i+1]) == pair:
           data.append(new_id)
           i += 2
       else:
           data.append(data_list[i])
           i += 1
    return data


def train(data_list, vocab_size):
    """Learn merges until the vocab reaches vocab_size. Returns (merges, merged data_list)."""
    merges = {}
    while len(chars) + len(merges) < vocab_size:
        new_id = len(chars) + len(merges)
        pair = findadjacent(data_list)
        merges[pair] = new_id
        data_list = merge(data_list,pair,new_id)
        if len(merges) % 50 == 0:
            print(f"merge {len(merges)}/{vocab_size - len(chars)}")
    return merges, data_list


def build_vocab(merges):
    """id -> string. Single chars first, then each merge = its two parts joined."""
    vocabs = dict(itos)
    for (a,b),new_id in merges.items():
        vocabs[new_id] = vocabs[a] + vocabs[b]
    return vocabs


def encode(text):
    """New text -> ids: start from chars, then replay every merge in the order it was learned."""
    data_list = encode_chars(text)
    for pair,new_id in merges.items():
        data_list = merge(data_list,pair,new_id)
    return data_list


def decode(ids):
    return "".join(vocabs[i] for i in ids)


def save(merges, data_list):
    # json can't use tuples as keys, so merges are saved as an ordered list of [a, b, new_id]
    with open(MERGES_PATH,'w') as f:
        json.dump([[a,b,new_id] for (a,b),new_id in merges.items()], f)
    with open(ENCODED_PATH,'w') as f:
        json.dump(data_list, f)


def load():
    """Return (merges, data_list) from disk, or None if missing or made with a different vocab_size."""
    if not (os.path.exists(MERGES_PATH) and os.path.exists(ENCODED_PATH)):
        return None
    with open(MERGES_PATH) as f:
        merges = {(a,b):new_id for a,b,new_id in json.load(f)}
    if len(chars) + len(merges) != vocab_size:
        return None
    with open(ENCODED_PATH) as f:
        data_list = json.load(f)
    return merges, data_list


def load_or_train():
    saved = load()
    if saved is not None:
        return saved
    print(f"No saved tokenizer for vocab_size {vocab_size}, training it (takes a few minutes)...")
    merges, data_list = train(encode_chars(data), vocab_size)
    save(merges, data_list)
    return merges, data_list


merges, data_list = load_or_train()
vocabs = build_vocab(merges)


def return_data():
    return data_list
