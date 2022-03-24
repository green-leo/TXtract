import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from config import cfg
from utils import EntityDataset
import engine
from models import TXtract_AVExtraction
import eval

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="utf-8")
    df.loc[:, "id"] = df["id"].fillna(method="ffill")
    df.loc[:, 'word'] = df['word'].fillna("")
    df = df[df['word']!=""]
    
    df.loc[~df["tag"].isin(cfg.TAGS), "tag"] = 'O'

    enc_tag = preprocessing.LabelEncoder()
    enc_tag.classes_ = np.array(cfg.TAGS)

    df.loc[:, "tag"] = enc_tag.transform(df["tag"])
    
    cates = df.groupby("id")["cate_id"].apply(list).values
    cates = [cate[0] for cate in cates]
    sentences = df.groupby("id")["word"].apply(list).values
    tags = df.groupby("id")["tag"].apply(list).values

    return cates, sentences, tags, enc_tag


def evaluate_model(valid_dataset, enc_tag):
    device = torch.device(cfg.DEVICE)
    model = TXtract_AVExtraction(cfg)
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    model.to(device)

    matrix = eval.compute_matrix(device, model, valid_dataset, enc_tag)
    return matrix


if __name__ == "__main__":
    
    ### Prepare data
    cates, sentences, tags, enc_tag = process_data(cfg.DATA_PATH)    
    meta_data = {
        "enc_tag": enc_tag
    }
    joblib.dump(meta_data, "meta.bin")

    (
        train_cates,
        test_cates,
        train_sentences,
        test_sentences,
        train_tags,
        test_tags
    ) = model_selection.train_test_split(cates, sentences, tags, random_state=42, test_size=0.1)

    train_dataset = EntityDataset(train_cates, train_sentences, train_tags, cfg)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )

    valid_dataset = EntityDataset(test_cates, test_sentences, test_tags, cfg)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.VALID_BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )

    ### Prepare model
    # configure model
    device = torch.device(cfg.DEVICE)
    model = TXtract_AVExtraction(cfg)
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.001,
    #     },
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]

    # optimizer = AdamW(optimizer_parameters, lr=cfg.LEARNING_RATE)
    # num_train_steps = int(len(train_sentences) / cfg.TRAIN_BATCH_SIZE * cfg.EPOCHS)
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    # )
    

    ### Training

    best_loss = np.inf
    for epoch in range(cfg.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        torch.save(model.state_dict(), 'last_model.bin')
        if test_loss < best_loss:
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            best_loss = test_loss



    ### Evaluating    
    matrix = evaluate_model(valid_dataset, enc_tag)
    print(matrix)