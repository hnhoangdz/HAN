import torch
import torch.nn as nn
from src.sent_attention import SentAttention
from src.word_attention import WordAttention


class HANnet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, word2vec_path, num_classes=10):
        super(HANnet, self).__init__()
        
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.batch_size = batch_size
        
        self.word_attn = WordAttention(word2vec_path, word_hidden_size)
        self.sent_attn = SentAttention(sent_hidden_size, word_hidden_size)
        
        self.fcn = nn.Linear(2*sent_hidden_size, num_classes)
        
        self._init_hidden_state()
                
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()
            
    def forward(self, x):
        # x: bs x n_sents x n_words
        word_out_list = []
        x = x.permute(1, 0, 2) # n_sents x bs x n_words
        for sent_batch in x:
            sent_batch = sent_batch.permute(1, 0) # n_words x bs
            print('input: ', sent_batch.shape)
            out, self.word_hidden_state = self.word_attn(sent_batch, self.word_hidden_state)
  
            word_out_list.append(out)
        
        # output of word attention model
        # high-representation of sentences in document
        out = torch.cat(word_out_list, 0) 
        print('out5: ', out.shape)
        # output of sentence attention model
        # high-representation of document
        out, self.sent_hidden_state = self.sent_attn(out, self.sent_hidden_state)
        
        # classifier layer 
        out = self.fcn(out)
        return out
        