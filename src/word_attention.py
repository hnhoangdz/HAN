import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
from src.utils import mul_element_wise, mul_matrix

class WordAttention(nn.Module):
    def __init__(self, dictionary_path, word_hidden_size=50):
        super(WordAttention, self).__init__()
        
        # get pre-trained dictionary model (ignore key-word)
        # original size: (vocab_size, embed_size)
        dictionary = pd.read_csv(filepath_or_buffer=dictionary_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        vocab_size, embed_size = dictionary.shape
        
        # add unknown word embedding as zeros in header
        vocab_size += 1
        unk_word = np.zeros((1, embed_size)) # make unk_word as index 0
        dictionary = torch.from_numpy(np.concatenate([unk_word, dictionary], axis=0).astype(np.float32))
        
        # parameters initialization
        self.word_weight = nn.Parameter(torch.Tensor(2 * word_hidden_size, 2 * word_hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * word_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * word_hidden_size, 1))
        
        # layer initializationf_output
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size).from_pretrained(dictionary)
        self.gru = nn.GRU(embed_size, word_hidden_size, bidirectional=True)
        self._create_weights()
        
    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
    
    def forward(self, x, hidden_state):

        out = self.embed_layer(x) # n_words x bs x embed_size
        f_out, h_out = self.gru(out.float(), hidden_state)
        # f_out: set of hidden states of whole sentences, whole words (n_words x bs x (2*word_hidden_size))
        # h_out: final hidden states of bidirectional (2 x bs x word_hidden_size)
        # print('f_out: ', f_out.shape)
        out = mul_matrix(f_out, self.word_weight, self.word_bias) # (n_words x bs x (2*word_hidden_size))
        # print('out1: ', out.shape)
        out = mul_matrix(out, self.context_weight, bias=False, is_context=True).permute(1,0) # (bs x n_words)
        # print('out2: ', out.shape)
        out = F.softmax(out, dim=1) # (bs x n_words)
        # print('out3: ', out.shape)
        out = mul_element_wise(f_out, out.permute(1,0)) # (1 x bs x (2*word_hidden_size))
        # print('out4: ', out.shape)    
        return out, h_out
        
if __name__ == '__main__':
    # dictionary_path = '../data/glove.6B.50d.txt'
    # x = torch.LongTensor([[1,2,4,5],
    #                       [4,3,2,9],
    #                       [1,2,4,5]])
    # h0 = torch.zeros(2, 4, 50)
    
    # word_att = WordAttention(dictionary_path)
    # out, h_out = word_att(x, h0)
    
    # dict = pd.read_csv(filepath_or_buffer=dictionary_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    # dict_len, embed_size = dict.shape
    # dict_len += 1
    # unknown_word = np.zeros((1, embed_size))
    # dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float32))
    # embed_layer = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
    # print('Input shape: ', x.shape)
    # out = embed_layer(x)
    # print('Output embedding shape: ', out.shape)
    # gru_layer = nn.GRU(50, 50, bidirectional=True)
    # f_out, h_out = gru_layer(out, h0)
    # print('F_output GRU shape: ', f_out.shape)
    # print('H_output GRU shape: ', h_out.shape)
    
    # a = nn.Parameter(torch.Tensor(1, 2)) 
    # print(a)
    # a = a.expand(2, 2)
    # print(a)
    x = torch.FloatTensor([[[1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12]],
                           
                           [[13,14,15,16],
                            [17,18,19,20],
                            [21,22,23,24]]])
    print(x)
    y = x.permute(1,0,2)
    z = x.transpose(0,1)
    print(y)
    print(z)
    print(x.T)