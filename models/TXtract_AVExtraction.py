import torch
import transformers
import torch.nn as nn
from .SeqSelfAttention import SeqSelfAttention
from torchcrf import CRF


class TXtract_AVExtraction(nn.Module):
    def __init__(self, cfg):
        super(TXtract_AVExtraction, self).__init__()
        self.use_cate = cfg.USE_CATE
        self.return_attention = cfg.RETURN_ATTENTION
        self.embedding_dim = cfg.EMBEDDING_DIM
        self.hidden_dim = cfg.HIDDEN_DIM
        self.tagset_size = cfg.TAGSET_SIZE

        self.bert = transformers.BertModel.from_pretrained(cfg.BASE_MODEL_PATH, return_dict=False)
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.seqselfattn = SeqSelfAttention(cfg, return_attention=self.return_attention)
        
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.hidden2tag = torch.nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)


    # return loss to train
    def forward(self, inputs):
        ec = inputs['cate']
        ids = inputs['ids']
        masks = inputs['mask']
        token_type_ids = inputs['token_type_ids']
        tags = inputs['target_tag']

        # embbeding        
        x, _ = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)
        
        # attention
        if self.use_cate:
            h_ = self.seqselfattn(h, ec)
        else:
            h_ = self.seqselfattn(h)
        
        if self.return_attention:
            h_ = h_[0]

        # crf
        logits = self.dropout(h_)
        logits = self.hidden2tag(logits)
        loss = - self.crf(logits, tags)

        return loss 


    # return pred_tags
    def encode(self, inputs):
        ec = inputs['cate']
        ids = inputs['ids']
        masks = inputs['mask']
        token_type_ids = inputs['token_type_ids']
        tags = inputs['target_tag']

        # embedding
        x, _ = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)
        
        # attention
        if self.use_cate:
            h_ = self.seqselfattn(h, ec)
        else:
            h_ = self.seqselfattn(h)
        
        if self.return_attention:
            h_, attn = h_
        
        # crf
        logits = self.dropout(h_)
        logits = self.hidden2tag(logits)
        outputs = self.crf.decode(logits)

        if self.return_attention:
            return outputs, attn
        
        return outputs 