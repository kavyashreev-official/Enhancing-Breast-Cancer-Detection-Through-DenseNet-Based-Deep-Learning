import cv2 
import os, sys
import random
import string
import os
import gc
import time
import mysql.connector as mssql
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import hashlib
import socket
global model
def getMachine_addr():
	os_type = sys.platform.lower()
	command = "wmic bios get serialnumber"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def getUUID_addr():
	os_type = sys.platform.lower()
	command = "wmic path win32_computersystemproduct get uuid"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def extract_command_result(key,string):
    substring = key
    index = string.find(substring)
    result = string[index + len(substring):]
    result = result.replace(" ","")
    result = result.replace("-","")
    return result
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.cuda.amp import GradScaler, autocast
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm
    import os
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for ImageNet models
    ])
    dataset = datasets.ImageFolder(root='Dataset', transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% training data
    val_size = len(dataset) - train_size  # 20% validation data
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load the pre-trained DenseNet201 model
    model = models.densenet201(pretrained=True)

    # Modify the classifier layer to match the number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

    # Move the model to the specified device (e.g., GPU)
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Set up the optimizer with all model parameters
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set up a learning rate scheduler to decay LR by 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    loader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    loader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data using tqdm to show progress
                for inputs, labels in tqdm(loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with autocast():  # Mixed precision training
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            print()

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)
    torch.save(model,',,/Model/model.pth')

def save_model():
    model = '../Model/model.pth'
    if os.path.exists(model):
        return True
    else:
        return False

    


def plot_accuracy():
    image = cv2.imread('../Plots/loss.png')
    return image

def plot_loss():
    image = cv2.imread('../Plots/loss.png')
    return image

def md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()

def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN

def key_validate(str):
    conn = mssql.connect(
        user='root', password='taylor@1989', host='localhost', database='hmp'
        )
    cur = conn.cursor()
    private_key = extract_command_result("SerialNumber",getMachine_addr()) + extract_command_result("UUID",getUUID_addr())
    if private_key in str:
        cur.execute("select * from SOFTKEY where private_key = %s and public_key = %s",(md5(private_key),md5(extract_command_result(private_key,str))))
        data=cur.fetchone()
        if data:
            return True
        else:
            return False
    else:
        return False

