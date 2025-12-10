"""
CS545 Final Project - Seismic Event Classification
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import d2l


# ______________________________________________________________________________
# Data Preparation

def collect_data(files_path):
    """
    Collect class-specific .csv files and create matrix X with the examples within them.
    """
    df_list = []
    total = 0
    for filename in files_path:
        print(f'Filename: {filename}, Number of Examples: {len(pd.read_csv(filename, header=None))}')
        total += len(pd.read_csv(filename, header=None))
        df_list.append(pd.read_csv(filename, header=None))
    print(f'Total number of examples: {total}')
    return pd.concat(df_list, axis=0).to_numpy()

class SeismicData(d2l.DataModule):
    """
    Seismic data class. Defines train/test/split and creates dataloaders.
    """
    def __init__(self, X, y, test_size=0.2, val_size=0.2, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.X = X
        self.y = y

        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size

        self.train, self.val, self.test = self.get_data_splits()
        

    def get_data_splits(self):
        X_trainval, X_test, y_trainval, y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y, shuffle=True, random_state=1)
        
        val_size = self.val_size / (1 - self.test_size) # adjust validation set size
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, shuffle=True, random_state=1)

        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.int64)
        y_val = torch.tensor(y_val, dtype=torch.int64)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        # Create Tensor Datasets
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test, y_test)

        return train_ds, val_ds, test_ds

    def get_dataloader(self, flag):
        shuffle = False
        if flag == 'train':
            data = self.train
            shuffle=True
        elif flag == 'val':
            data = self.val
        elif flag == 'test':
            data = self.test
        else:
            raise Exception("flag must be 'train', 'val', or 'test'")
        
        return DataLoader(data, self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_class_names(self):
        class_names = np.unique(self.y)
        return class_names

def get_random_waveforms(data, size=3):
    """
    Get random seismic waveforms, where the variable `size` sets the number of samples fetched per class.
    """
    waveform_dict = {}
    class_names = data.get_class_names()
    for label in class_names:
        label_waveforms = data.X[data.y == label] 
        random_indices = np.random.choice(len(label_waveforms), size=size, replace=False)
        
        waveform_dict[label] = label_waveforms[random_indices]
    return waveform_dict

def plot_waveforms(data, waveform_dict):
    """
    Plots seismic waveforms. Each row is a separate class, and each column is a seismic waveform.
    """
    nrows=len(data.get_class_names()) # total number of classes
    ncols=len(waveform_dict[data.get_class_names()[0]]) # total number of waveforms per class
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize = (ncols * 3, nrows * 3))
    for i, label in enumerate(waveform_dict):
        for j, waveform in enumerate(waveform_dict[label]):
            ax[i, j].plot(waveform)
            ax[i, j].set_xlabel("Time")
            ax[i, j].set_ylabel("Amplitude")
        ax[i, ncols // 2].set_title(f'Class {label.astype(int)}', fontsize=20)
    plt.tight_layout()
    plt.show()

def display_class_distribution(class_names, counts):
    """
    Plots a bar graph with the distribution of seismic classes.
    """
    colors = plt.cm.magma(np.linspace(0, 1, len(class_names)))
    
    plt.figure(figsize=(10, 5))
    plt.bar(class_names.astype(int), counts, color=colors)
    plt.title("Distribution of Seismic Waves")
    plt.xlabel("Seismic Wave Types")
    plt.xticks(class_names.astype(int))
    plt.ylabel("Number of Samples")
    plt.show()


# ______________________________________________________________________________
# Model Development

def evaluation_metrics(model, data):
    """
    Outputs the following evaluation metrics: accuracy, precision, recall, f1-score, and a confusion matrix
    """
    actual = []
    predicted = []
    
    model.eval()
    for batch_X, batch_y in data.get_dataloader('train'):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        with torch.no_grad():
            preds = model(batch_X)

            y_actual = batch_y.cpu().tolist()
            y_pred = torch.argmax(preds, dim=1).cpu().tolist()

            actual.extend(y_actual)
            predicted.extend(y_pred)
    accuracy = accuracy_score(actual, predicted) 
    report = classification_report(actual, predicted, 
                                   target_names=data.get_class_names(), zero_division=0)
    confusion_matrix = confusion_matrix(actual, predicted)

    results = {
        'accuracy': accuracy,
        'report': report,
        'confusion matrix': confusion_matrix
    }
    return results

def display_confusion_matrix(confusion_matrix, class_names):
    """
    Displays confusion matrix.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(actual, predicted), display_labels=class_names)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax, include_values=True, cmap=plt.cm.Blues)
    ax.set_xticklabels(class_names, rotation=90)

def train(model, data, optimizer, epochs, device):
    """
    Training loop for models.
    """
    loss_fn = nn.CrossEntropyLoss()
    
    loss_train = []
    loss_valid = []
    accuracy_train = []
    accuracy_valid = []
    
    for epoch in range(epochs):
        print("epoch", epoch + 1,)
        model.train()
        loss_values = [] # loss values for each batch
        accuracy_values = [] # accuracy values for each batch
        for batch_X, batch_y in data.get_dataloader('train'):
            batch_X, batch_y = batch_X.unsqueeze(1).to(device), batch_y.to(device)
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            loss_values.append(loss.item())
            accuracy_values.append(model.accuracy(preds, batch_y).item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train.append(np.mean(loss_values))
        accuracy_train.append(np.mean(accuracy_values))
        
        model.eval()
        loss_values = []
        accuracy_values = []
        
        for batch_X, batch_y in data.get_dataloader('val'):
            batch_X, batch_y = batch_X.unsqueeze(1).to(device), batch_y.to(device)
            with torch.no_grad():
                preds = model(batch_X)
                loss = loss_fn(preds, batch_y)
                loss_values.append(loss.item())
                accuracy_values.append(model.accuracy(preds, batch_y).item())
                
        loss_valid.append(np.mean(loss_values))
        accuracy_valid.append(np.mean(accuracy_values))
        print ("accuracy: ", accuracy_valid[-1])
    return loss_train, loss_valid, accuracy_train, accuracy_valid

def init_xavier(module):
    """
    Defines xavier weight initialization.
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d:
        nn.init.xavier_uniform_(module.weight)

def init_kaiming(module):
    """
    Defines kaiming weight initialization.
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d:
        nn.init.kaiming_uniform_(module.weight)