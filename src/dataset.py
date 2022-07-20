import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import torch

""""
AG News Classification Dataset

Training: 120,000 samples (train.csv)
Testing: 7,600 samples (test.csv)
#Classes: 4 (World, Sports, Business, Sci/Tech) (classes.txt)

Each sample contains 3 columns: class index (1 - 4), title, description
"""

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentence=30, max_length_word=35):
        super(MyDataset, self).__init__()
        
        texts, labels = [], []
        
        with open(data_path) as csv_file:
            
            reader = csv.reader(csv_file, quotechar='"')
            
            for _, line in enumerate(reader):
                # merge title and description as a document
                text = ""
                for txt in line[1:]:
                    text += txt.lower()
                    text += " "
                label = int(line[0]) - 1
                
                texts.append(text) # list of documents
                labels.append(label) # list of corresponding labels
        
        self.texts = texts
        self.labels = labels
        
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentence = max_length_sentence
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        label = self.labels[index]
        text = self.texts[index]
        # print(text)
        doc2idx = [[self.dict.index(word) if word in self.dict else -1 
                    for word in word_tokenize(sentences)] 
                        for sentences in sent_tokenize(text=text)]
        # print(len(doc2idx))
        # extend more words in sentence to equal to max_length_word
        for sen2idx in doc2idx:
            # print(1)
            if len(sen2idx) < self.max_length_word:
                extend_word_idx = [-1]*(self.max_length_word - len(sen2idx))
                sen2idx.extend(extend_word_idx)
                
        # extend more sentences in document to equal to max_length_sentence
        if len(doc2idx) < self.max_length_sentence:
            extend_sen_idx = [[-1]*self.max_length_word]*(self.max_length_sentence - len(doc2idx))
            doc2idx.extend(extend_sen_idx)
        
        # ensure shape of a document like: (30 x 35) - a docuement has 30 sentences, each sentence has 35 words 
        doc2idx = [sen2idx[:self.max_length_word] for sen2idx in doc2idx[:self.max_length_sentence]]
        
        doc2idx = np.stack(arrays=doc2idx, axis=0)
        doc2idx += 1
        
        return doc2idx.astype(np.int64), label
        
if __name__ == '__main__':
    data_path = '../data/ag_news_csv/train.csv'
    dict_path = '../data/glove.6B.50d.txt'
    dataset = MyDataset(data_path, dict_path)
    print(dataset[1][0].shape)
    
    x = torch.FloatTensor([[1,2],
                           [3,4]])