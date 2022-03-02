# coding:utf8
from pickle import FALSE
import warnings
import torch
import os
from transformers import AutoTokenizer
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME


class DefaultConfig(object):
    ENV = 'default'                 # visdom 
    VIS_PORT = 8097                 # visdom
    
    MODEL_NAME = 'TXtract'
    USE_CATE = False
    RETURN_ATTENTION = True

    PRETRAINED_BERT_NAME = 'phobert-base'
    PRETRAINED_MODEL_DIR = './pretrained_model'
    BASE_MODEL_PATH = os.path.join(PRETRAINED_MODEL_DIR, PRETRAINED_BERT_NAME)
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, do_lower_case=True)
    
    DATA_PATH = './data/product_title_dataset.csv'
    pickle_path = '???'
    load_model_path = None          # trained model

    TRAIN_BATCH_SIZE = 32                 # batch size
    VALID_BATCH_SIZE = 8
    MAX_LEN = 256
    EMBEDDING_DIM = 768             # Bert
    HIDDEN_DIM = 1024               # BiLSTM
    ATTN_DIM = 32                   # SeqSelfAttention
    CATE_DIM = 1                    # Poincare Ball

    LABELS = ['type', 'name']
    TAGS = [bie + '-' + label for label in LABELS for bie in ['B', 'I', 'E']]
    TAGS.append('O')
    # TAGSET_SIZE = len(LABELS)*3 + 1 # BIOE tagging, 3-B,I,E for each label, 1-O for other
    TAGSET_SIZE = len(TAGS)

    DEVICE = "cpu"
    USE_GPU = False

    NUM_WORKERS = 0                 # how many workers for loading data,  # cpu only, 0 for easy loading
    PRINT_FREQ = 100                # print info every N batch

    EPOCHS = 20
    LEARNING_RATE = 2e-5            # initial learning rate
    LR_DECAY = 0.5                  # when val_loss increase, lr = lr*lr_decay
    WEIGHT_DECAY = 0e-5             # L2 norm
    DROPOUT = 0.2
    SEED = 1234

    MODEL_PATH = 'best_model.bin'


    def _parse(self, kwargs):
        """ 
        Update config parameters according to dictionary kwargs
        """
        
        # Update parameters
        for k, v in kwargs.items():
            if not hasattr(self, k.upper()):
                warnings.warn("Warning: cfg has not attribute %s" % k.upper())
            setattr(self, k.upper(), v)
        
        # Check device
        cfg.DEVICE = torch.device('cuda') if cfg.USE_GPU else torch.device('cpu')

        # Final user config
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
                
    # end def

# end class


cfg = DefaultConfig()
