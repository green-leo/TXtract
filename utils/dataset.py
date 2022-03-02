from numpy import dtype
import config
import torch

def get_special_token_ids(tokenizer):
    special_token_id = tokenizer.encode('')
    cls_id = special_token_id[0]
    sep_id = special_token_id[1]
    return cls_id, sep_id


class EntityDataset:
    def __init__(self, cates, texts, tags, cfg):
        # cates: m-dim for category of the product in title [cate1, cate2, ....]
        # texts: ['Toddler and Baby Girls' Socks with Non-Skid Soles, Pack of 12'.split(), 
        #         'Blink Mini – Compact indoor plug-in smart security camera, 1080 HD video, ....'.split(),
        #          ...]
        # tags: [[B-Type, I-Type, E-Type, O, O, B-?,...], [...], ...]]
        self.cates = cates
        self.texts = texts
        self.tags = tags
        self.tokenizer = cfg.TOKENIZER
        self.max_len = cfg.MAX_LEN
        self.cls_id, self.sep_id = get_special_token_ids(self.tokenizer)

    
    def __len__(self):
        return len(self.cates)
    
    def __getitem__(self, item):
        cate = self.cates[item]
        text = self.texts[item]
        tag = self.tags[item]

        ids = []
        target_tag =[]
        
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s,
                add_special_tokens=False
            )
            # a word can be encoded to one or many ids
            # specially for other language word

            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tag[i]] * input_len)

        ids = ids[:self.max_len - 2]
        target_tag = target_tag[:self.max_len - 2]

        ids = [self.cls_id] + ids + [self.sep_id]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "cate": torch.tensor(cate, dtype=torch.float),
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
