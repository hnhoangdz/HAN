import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import mul_element_wise, mul_matrix

class SentAttention(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        super(SentAttention, self).__init__()
        
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.sent_context = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
        
        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        
        self._create_weights()
        
    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.sent_context.data.normal_(mean, std)
        
    def forward(self, x, hidden_state):
        
        f_out, h_out = self.gru(x, hidden_state)
        print('f_out: ', f_out.shape)
        out = mul_matrix(f_out, self.sent_weight, self.sent_bias)
        print('out: ', f_out.shape)
        out = mul_matrix(f_out, self.sent_context, is_context=True).permute(1,0)
        print('out1: ', f_out.shape)
        out = F.softmax(out, dim=1)
        print('out2: ', f_out.shape)
        out = mul_element_wise(f_out, out.permute(1,0)).squeeze(0)
        print('out3: ', f_out.shape)
        return out, h_out
    
if __name__ == '__main__':
    x = torch.rand(2, 128, 100)
    hs = torch.rand(2, 128, 50)
    sent_attn = SentAttention()
    out,h = sent_attn(x,hs)
    print(out.shape)