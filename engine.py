import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        
        #CRF
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3)
        optimizer.step()
        # scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model(data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
