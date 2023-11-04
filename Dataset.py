import numpy as np
import torch

class SequenceDataset(torch.utils.data.Dataset):
    
    def __init__(self, word2id, fam2label, max_len, data_path, split, u):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.data, self.label = u.reader(split, data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
       
        return {'sequence': seq, 'target' : label}
    
    def preprocess(self, text):
        seq = []
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>']] * (self.max_len - len(seq))
        seq = torch.tensor(seq, dtype=torch.long)
        return seq
