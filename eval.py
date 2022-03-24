import torch

from config import cfg
from sklearn import metrics

def classification_report(y_pred, y_test, labels):
    return (metrics.classification_report(
        y_test, y_pred, labels=labels, digits=3
    ))


def get_prediction_tag(device, model, valid_dataset, enc_tag):
    tags = []
    raw_tags = []
    with torch.no_grad():
        # data = valid_dataset
        for data in valid_dataset:
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag = model.encode(data)
            raw_tags.append(tag[0])
            # print(tag)
            tag = enc_tag.inverse_transform(
                    tag[0]
                )
            tags.append(tag.tolist())

    return raw_tags, tags


def get_corresponding_target_tag(tags, valid_dataset, enc_tag):
    target_tags = []
    raw_target_tags = []

    for ii, data in enumerate(valid_dataset):
        # print(data)
        target_tag = data['target_tag'][:len(tags[ii])]
        raw_target_tags.append(target_tag.tolist())
        target_tags.append(enc_tag.inverse_transform(target_tag).tolist())

    return target_tags, raw_target_tags


def compute_matrix(device, model, valid_dataset, enc_tag):
    raw_tags, tags = get_prediction_tag(device, model, valid_dataset, enc_tag)
    target_tags, raw_target_tags = get_corresponding_target_tag(tags, valid_dataset, enc_tag)

    flat_tags = [item for sublist in raw_tags for item in sublist]
    flat_target_tags = [item for sublist in raw_target_tags for item in sublist]

    matrix = classification_report(
        enc_tag.inverse_transform(flat_tags), 
        enc_tag.inverse_transform(flat_target_tags),
        labels=cfg.TAGS
    )

    return matrix