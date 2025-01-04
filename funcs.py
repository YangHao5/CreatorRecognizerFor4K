'''
funcs.py
Defining functions used for preprocessing, importing & saving data,
splitting dataset, training, validating & training, etc.
DO NOT RUN THIS FILE DIRECTLY.
'''
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import pandas as pd

def import_tensors(directory_path):
    files = [str(file) for file in Path(directory_path).rglob('*.pt')]
    notes, creators, cids = [], [], []
    for file in files:
        chart = torch.load(file, weights_only=True)
        notes.append(chart['notes'])
        creators.append(chart['creator'])
        cids.append(torch.tensor(chart['cid']))
    notes = torch.stack(notes)
    cids = torch.stack(cids)
    print(f'Imported {len(cids)} files.')
    # [chart_num, 3, 4, 1000], len(chart_num)
    return notes, creators, cids

def augment_data(notes, creators):
    augmented_notes = []
    augmented_creators = []
    
    for i in range(len(notes)):
        if creators[i] != 0:
            original = notes[i].unsqueeze(0) if isinstance(notes[i], torch.Tensor) else torch.tensor(notes[i]).unsqueeze(0)
            creator = creators[i].unsqueeze(0) if isinstance(creators[i], torch.Tensor) else torch.tensor(creators[i]).unsqueeze(0)

            if original.shape[2] >= 4:
                flipped = original.clone()
                flipped = flipped[:, :, [3, 2, 1, 0], :]

                augmented_notes.append(flipped)
                augmented_creators.append(creator)

    if augmented_notes:
        augmented_notes = torch.cat(augmented_notes, dim=0)
        augmented_creators = torch.cat(augmented_creators, dim=0)
        return augmented_notes, augmented_creators
    else:
        return None, None

def split(notes, creators, cids, train_ratio, batch_size=32):
    count = Counter(creators)
    target_creators = [s for s, freq in count.items() if freq >= 20]
    unique_creators = sorted(list(set(target_creators)))
    creator_to_index = {s: i+1 for i, s in enumerate(unique_creators)}
    
    creators_tensor = torch.zeros(len(creators), dtype=int)
    for i, s in enumerate(creators):
        if s in target_creators:
            creators_tensor[i] = creator_to_index[s]
        else:
            creators_tensor[i] = 0

    notes = notes.float()
    chart_num = notes.shape[0]
    train_size = int(train_ratio * chart_num)

    indices = torch.randperm(chart_num)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_notes, val_notes = notes[train_indices], notes[val_indices]
    train_creators, val_creators = creators_tensor[train_indices], creators_tensor[val_indices]

    augmented_notes, augmented_creators = augment_data(train_notes, train_creators)
    ori_num = len(creators)
    aug_num = 0
    if augmented_notes is not None:
        aug_num = augmented_creators.shape[0]
        train_notes = torch.cat([train_notes, augmented_notes], dim=0)
        train_creators = torch.cat([train_creators, augmented_creators], dim=0)
    print(f'{ori_num + aug_num} files in total. {ori_num} origin, {aug_num} generated.')
    print(f'Dataset split info: train {train_creators.shape[0]}, val {val_creators.shape[0]}.')

    train_dataset = TensorDataset(train_notes, train_creators)
    val_dataset = TensorDataset(val_notes, val_creators)
    whole_dataset = TensorDataset(notes, creators_tensor, cids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    whole_loader = DataLoader(whole_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, whole_loader, creator_to_index

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_targets, all_preds)
    pre = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, pre, rec

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_targets, all_preds)
    pre = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, pre, rec

def train_val(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    best_acc = 0.
    best_model_state = model.state_dict()
    for e in range(epochs):
        train_loss, train_acc, train_pre, train_rec = train(model, train_loader, criterion, optimizer, device)
        if (e + 1) % 10 == 0:
            print(f'Epoch {e+1}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Pre: {train_pre:.4f}, Rec: {train_rec:.4f}')
            val_loss, val_acc, val_pre, val_rec = evaluate(model, val_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Pre: {val_pre:.4f}, Rec: {val_rec:.4f}')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict()
    return best_model_state

def test(model, test_loader, device, cids, creator_to_index, csv_path):
    model.eval()
    odds, cids = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, cids_batch = batch[0].to(device), batch[2]
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            odds.append(probabilities.cpu().numpy())
            cids.extend(cids_batch.cpu().numpy())

    odds = np.concatenate(odds, axis=0)  # [chart_num, num_classes]
    cids = np.array(cids).reshape(-1, 1)

    index_to_creator = {v: k for k, v in creator_to_index.items()}
    column_names = ['cid', 'Others'] + [index_to_creator.get(i, f'Class_{i}') for i in range(1, odds.shape[1])]

    odds_with_cid = np.hstack((cids, odds))  # [chart_num, num_classes+1]

    odds_df = pd.DataFrame(odds_with_cid, columns=column_names)
    odds_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'Odds saved to {csv_path}.')

