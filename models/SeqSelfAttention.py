from turtle import ht
import torch
import transformers
import torch.nn as nn


class SeqSelfAttention(nn.Module):
    '''
    g(t,t_) = tanh(W1ht + W2ht_ + W3ec + bg)
    α(t,t_) = σ(wTα x g(t,t_) + bα), t, t_ = 1..T
    α(t,t_) = softmax(wTα x g(t,t_) + bα)
    h_(t) = Σ(t_=1->T) α(t,t_) · ht_

    '''
    
    def __init__(self, cfg, return_attention=False):
        super(SeqSelfAttention, self).__init__()
        self.device = torch.device(cfg.DEVICE)
        self.return_attention = return_attention
        self.d = cfg.HIDDEN_DIM
        self.p = cfg.ATTN_DIM

        self.W1 = nn.Linear(self.d, self.p)
        self.W2 = nn.Linear(self.d, self.p)
        
        if cfg.USE_CATE:
            self.m = cfg.CATE_DIM
            self.W3 = nn.Linear(self.m, self.p)
        else:
            self.m = None
            self.W3 = None

        self.wa = nn.Linear(self.p, 1)


    # batch size ?????
    def forward(self, inputs, ec=None):
        # inputs [batch_size, t, d]
        # ht: input [1 x t x d]
        # t: len input (number of tokens)
        # d: hidden_dim (lstm)
        # p: att_dim (32 by default)

        inputs_shape = inputs.shape
        batch_size = inputs_shape[0]
        t = inputs_shape[1]
        
        # g(t,t_) = tanh(W1ht + W2ht_ + W3ec + bg)
        W1Ht = self.W1(inputs)                                      # [batch_size, t, p]
        W1Ht = torch.unsqueeze(W1Ht, 2)                             # for W1ht + W2ht [batch_size, t, 1, p]

        W2Ht = self.W2(inputs)                                      # [batch_size, t, p]
        W2Ht = torch.unsqueeze(W2Ht, 1)                             # for W1ht + W2ht [batch_size, 1, t, p]

        W3ec = torch.zeros(batch_size, self.p).to(self.device)
        if ec != None:
            W3ec = self.W3(ec)                                      # [batch_size, p]
        
        # print(W1Ht.shape, W2Ht.shape, W3ec.shape)
        W3ec = W3ec.tile(1, t * t)                                         # [batch_size*1, p*t*t]
        # print(W3ec.shape)
        W3ec = W3ec.reshape(batch_size, t, t, self.p)                      # expand w3ec to [txt], no batch_sum provided

        # g(t,t_)
        Gt = torch.tanh(W1Ht + W2Ht + W3ec)                         # [batch_size, t, t, p]

        # α(t,t_) = σ(wTα x g(t,t_) + bα), t, t_ = 1..T
        # α(t,t_) = softmax(wTα x g(t,t_) + bα) 
        At = self.wa(Gt)                                            # [batch_size, t, t, 1]
        At = At.reshape(batch_size, t, t)                           # [batch_size, t, t]
        
        At = torch.softmax(At, dim=-1)                                      # [batch_size, t, t]
        # At = torch.sigmoid(At)
        
        # h_(t) = Σ(t_=1->T) α(t,t_) · ht_
        Ht_ = torch.bmm(At, inputs)                                 # [batch_size, t, d]

        if self.return_attention:
            return Ht_, At

        return Ht_