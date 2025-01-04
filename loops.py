'''
loops.py
Defining training & validating loop and testing loop.
DO NOT RUN THIS FILE DIRECTLY.
'''
import torch
import torch.optim as optim
from model import *
from funcs import *

def Train_and_Validate_Loop(tensor_path, device, epochs, model_path):
    notes, creators, cids = import_tensors(directory_path=tensor_path)
    train_loader, val_loader, _, creator_to_index = split(notes=notes, creators=creators, cids=cids,
                                                          train_ratio=0.7, batch_size=32)

    num_classes = 1 + len(creator_to_index)
    model = CreatorRecognizerFor4K(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_model_state = train_val(model, train_loader, val_loader, criterion, optimizer, 
                           epochs=epochs, device=device)
    torch.save(best_model_state, model_path)
    print(f'Best model saved to {model_path}.')

def Test_Loop(tensor_path, device, model_path, csv_path):
    notes, creators, cids = import_tensors(directory_path=tensor_path)
    _, _, whole_loader, creator_to_index = split(notes=notes, creators=creators, cids=cids,
                                                 train_ratio=0.7, batch_size=32)

    num_classes = 1 + len(creator_to_index)
    model = CreatorRecognizerFor4K(num_classes=num_classes).to(device)

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f'Model loaded from {model_path}.')
    else:
        print(f'Model file {model_path} not found. Run training first.')
        return

    test(model, whole_loader, device, cids, creator_to_index, csv_path)
