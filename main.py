'''
main.py
Running training & validating and testing loops.
Usage: python main.py
'''
from model import *
from funcs import *
from loops import *

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device used
dataset_path = 'processed_tensors'  # Tensor folder path used for training & validating
test_path = dataset_path    # Tensor folder path used for testing
epochs = 200    # Training Epochs. Validate once per 10 training epochs
model_path = 'model.pth'    # Model saving path
csv_path = 'res.csv'    # Result .csv file path

print(f'Device = {device}.')

# Train and validate
Train_and_Validate_Loop(tensor_path=dataset_path, device=device, epochs=epochs, model_path=model_path)

# Test
# You need to train or prepare a model .pth file before testing
Test_Loop(tensor_path=test_path, device=device, model_path=model_path, csv_path=csv_path)