import joblib
import torch

from config import cfg
from utils import EntityDataset
from models import TXtract_AVExtraction

import argparse

def encode_sentence(model, sentence):
    tokenized_sentence = cfg.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = EntityDataset(
        cates=[1428],
        texts=[sentence],  
        tags=[[0] * len(sentence)],
        cfg=cfg
    )
    
    device = torch.device(cfg.DEVICE)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag = model.encode(data)
        # print(tag)

    return tag



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a sentence.')
    parser.add_argument('sentence', metavar='String', type=str, nargs='+',
                        default = 'a default string', help='a sentence for prediction')
    
    args = parser.parse_args()
    sentence = getattr(args, 'sentence')
    sentence = ' '.join(sentence)

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    
    device = torch.device(cfg.DEVICE)
    model = TXtract_AVExtraction(cfg)
    model.load_state_dict(torch.load('best_model.bin'))
    model.to(device)

    tag = encode_sentence(model, sentence)
    tag = enc_tag.inverse_transform(tag[0])

    print(tag)
    
