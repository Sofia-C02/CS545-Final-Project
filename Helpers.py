from matplotlib import pyplot as plt
import torch 
from torch import nn
import numpy as np
import pandas as pd
from d2l import torch as d2l
import random
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

###############################
class explore:

    def display_class_distribution(class_names, counts):
        colors = plt.cm.magma(np.linspace(0, 1, len(class_names)))
        
        plt.figure(figsize=(10, 5))
        plt.bar(class_names.astype(int), counts, color=colors)
        plt.title("Distribution of Seismic Waves")
        plt.xlabel("Seismic Wave Types")
        plt.xticks(class_names.astype(int))
        plt.ylabel("Number of Samples")
        plt.show()

    def get_random_waveforms(data, size=3):
        waveform_dict = {}
        class_names = data.get_class_names()
        for label in class_names:
            label_waveforms = data.X[data.y == label] # assuming data class has X, y as member variables
            random_indices = np.random.choice(len(label_waveforms), size=size, replace=False)
            
            waveform_dict[label] = label_waveforms[random_indices]
        return waveform_dict


    def plot_waveforms(data, waveform_dict):
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

###############################
class evaluate:

    def evaluation_metrics(model, data):
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
        accuracy = balanced_accuracy_score(actual, predicted) # determine if to use balanced_accuracy or just accuracy
        report = classification_report(actual, predicted, 
                                       target_names=data.get_class_names(), zero_division=0) # make a get_class_names function to get class names
        confusion_matrix = confusion_matrix(actual, predicted)
    
        results = {
            'accuracy': accuracy,
            'report': report,
            'confusion matrix': confusion_matrix
        }
        return results

    def display_confusion_matrix(confusion_matrix, class_names):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(actual, predicted), display_labels=class_names)
        fig, ax = plt.subplots(figsize=(15, 15))
        disp.plot(ax=ax, include_values=True, cmap=plt.cm.Blues)
        ax.set_xticklabels(class_names, rotation=90)
    def loss_curve(loss_train, loss_valid, accuracy_valid):
        epochs = range(1, len(loss_train)+1)
        
        # plot loss curve
        fig, ax = plt.subplots(1, figsize=(5,3))
        l1, = ax.plot(epochs, loss_train, 'skyblue', label='training loss')
        l2, = ax.plot(epochs, loss_valid, 'deepskyblue', label='validation loss')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.yaxis.label.set_color('lightskyblue')

        ax2 = ax.twinx()
        l3, = ax2.plot(epochs, accuracy_valid, 'indianred', label='validation accuracy')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.yaxis.label.set_color('indianred')
        
        lines = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper center')
        ax.set_xlabel('Epoch', fontweight='bold')
        plt.tight_layout()
        plt.show()

##############################
class SeismicData(d2l.DataModule):
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
