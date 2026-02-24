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

# At top of controller.py
training_progress = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 10,
    "batch": 0,
    "total_batches": 0,
    "percent": 0
}

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
training_progress = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 10,
    "batch": 0,
    "total_batches": 0,
    "percent": 0
}

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

    global training_progress
    training_progress["status"] = "running"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root='../Dataset', transform=transform)

    # Train-val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = models.densenet201(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler()

    num_epochs = 10
    training_progress["total_epochs"] = num_epochs

    def train_model(model, train_loader, val_loader):
        total_steps = num_epochs * len(train_loader)

        global training_progress

        for epoch in range(num_epochs):
            training_progress["epoch"] = epoch + 1

            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                    loader = train_loader
                else:
                    model.eval()
                    loader = val_loader

                training_progress["total_batches"] = len(loader)

                for batch_i, (inputs, labels) in enumerate(tqdm(loader)):
                    training_progress["batch"] = batch_i + 1

                    # Update percent
                    completed_steps = (epoch * len(train_loader)) + batch_i + 1
                    training_progress["percent"] = int((completed_steps / total_steps) * 100)

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                if phase == 'train':
                    scheduler.step()

        training_progress["status"] = "completed"
        training_progress["percent"] = 100

    # Run training
    train_model(model, train_loader, val_loader)

    # Save the model
    torch.save(model, '../Model/model.pth')

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

