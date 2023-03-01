import transformers
import torch
import torch.nn as nn
from torchcrf import CRF
from .AttributeAttention import AttributeAttention
from .squeeze_embedding import SqueezeEmbedding


class SUOpenTag(nn.Module):
    def __init__(self, cfg):
        super(SUOpenTag, self).__init__()

        self.return_attention = cfg.RETURN_ATTENTION

        self.embedding_dim = cfg.EMBEDDING_DIM
        self.hidden_dim = cfg.HIDDEN_DIM
        self.tagset_size = cfg.TAGSET_SIZE

        self.bert = transformers.BertModel.from_pretrained(cfg.BASE_MODEL_PATH, return_dict=False)
        self.t_bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.a_bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.attrattn = AttributeAttention()
        self.layernorm = nn.LayerNorm(self.hidden_dim* 2)

        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.hidden2tag = torch.nn.Linear(self.hidden_dim*2, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)



    # return loss log likelihood
    def forward(self, inputs):
        t_ids = inputs['t_ids']
        t_masks = inputs['t_mask']
        t_token_type_ids = inputs['t_token_type_ids']
        a_ids = inputs['a_ids']
        a_masks = inputs['a_mask']
        a_token_type_ids = inputs['a_token_type_ids']
        tags = inputs['target_tag']

        # embbeding        
        t_w, _ = self.bert(t_ids, attention_mask=t_masks, token_type_ids=t_token_type_ids)
        t_h, _ = self.bilstm(t_w)                   # hidden states as context              # N x L x (D x H)

        a_w, _ = self.bert(a_ids, attention_mask=a_masks, token_type_ids=a_token_type_ids)
        _, a_h = self.bilstm(a_w)                   # just take last hiddent state          # N x (D x 1) x H
        
        #a_h = torch.cat([a_h[0][-2],a_h[0][-1]], dim=-1)
        a_h = a_h.view(-1, self.hidden_dim)         # N x (D x H)
        
        # attention
        t_c, _ = self.attrattn(t_h, a_h)
        t_m = torch.cat((t_h, t_c), dim=-1)
        t_m = self.layernorm(t_m)

        # crf
        logits = self.dropout(t_m)
        logits = self.hidden2tag(logits)
        loss = - self.crf(logits, tags)

        return loss 



    # return pred_tags
    def encode(self, inputs):
        t_ids = inputs['t_ids']
        t_masks = inputs['t_mask']
        t_token_type_ids = inputs['t_token_type_ids']
        a_ids = inputs['a_ids']
        a_masks = inputs['a_mask']
        a_token_type_ids = inputs['a_token_type_ids']
        tags = inputs['target_tag']

        # embbeding        
        t_w, _ = self.bert(t_ids, attention_mask=t_masks, token_type_ids=t_token_type_ids)
        t_h, _ = self.bilstm(t_w)                   # hidden states as context              # N x L x (D x H)

        a_w, _ = self.bert(a_ids, attention_mask=a_masks, token_type_ids=a_token_type_ids)
        _, a_h = self.bilstm(a_w)                   # just take last hiddent state          # N x (D x 1) x H
        
        #a_h = torch.cat([a_h[0][-2],a_h[0][-1]], dim=-1)
        a_h = a_h.view(-1, self.hidden_dim)         # N x (D x H)
        
        # attention
        t_c, attn = self.attrattn(t_h, a_h)
        t_m = torch.cat((t_h, t_c), dim=-1)
        t_m = self.layernorm(t_m)

        # crf
        logits = self.dropout(t_m)
        logits = self.hidden2tag(logits)
        outputs = self.crf.decode(logits)

        return outputs, attn 

