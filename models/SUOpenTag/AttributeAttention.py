import torch.nn as nn
import torch


class AttributeAttention(nn.Module):
    def __init__(self):
        super(AttributeAttention,self).__init__()

    def forward(self, t_h, a_h):
        '''
        t_h : title_output  N x L x (D x H)           (batchsize, seqlen, hidden_dim)
        a_h : attr_output   N x (D x H)               (batchsize, hidden_dim)
        '''

        seq_len = t_h.size()[1]
        a_h = a_h.unsqueeze(1).repeat(1, seq_len, 1)

        attn = torch.cosine_similarity(t_h, a_h, -1)    # N x L
        attn = attn.unsqueeze(-1)                       # N x L x 1
        
        t_c = t_h * attn
        
        return t_c, attn
