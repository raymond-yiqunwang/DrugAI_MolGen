import os
import torch
import pandas as pd

class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
    
    def __len__(self):
        return len(self.idx2char)


class Dataset(object):
    def __init__(self, data_root):
        self.dictionary = Dictionary()
        self.dataset = self.tokenize(data_root)

    def tokenize(self, data_root):
        assert(os.path.isfile(data_root))
        with open(data_root, 'r') as f:
            idss = []
            for line in f:
                line = line[:-1] # get rid of '\n'
                ids = []
                for char in line:
                    self.dictionary.add_char(char)
                    ids.append(self.dictionary.char2idx[char])
                idss.append(torch.tensor(ids).type(torch.int64))
            idss = torch.cat(idss)
        
        return idss


