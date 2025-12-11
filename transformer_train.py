#!pip install d2l==1.0.3 --no-deps
import Transformer as T
import Helpers as H
import torch
from torch import nn
from d2l import torch as d2l
from glob import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        device = torch.device("cuda")
else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
# ---------------------------------
# Data management
# ---------------------------------
print("Loading data.")
filename = 'full_dataset.csv'
dataset_df = pd.read_csv(filename, header=None, dtype=float, on_bad_lines='skip')
dataset_df.dropna( axis=1,inplace=True)
dataset_df[0] = dataset_df[0].astype(int)
dataset_df = dataset_df.iloc[:,:2000]
print("Data loaded.")

dataset_np = dataset_df.to_numpy()
dataset_np.shape, dataset_df.shape
X, y = dataset_np[:, 1:], dataset_np[:, 0]
X = (X ) / X.std(axis=1, keepdims=True)

print(f"Classes in dataset:\n{dataset_df.iloc[:,0].unique()}")
dataset = H.SeismicData(X, y, test_size=0.2, val_size=0.2, batch_size=32)

print(f"Total samples: {len(dataset.y)}")
print(f"Train samples: {len(dataset.train)}")
print(f"Test samples: {len(dataset.test)}")

class_names, counts = np.unique(dataset.y, return_counts=True)

# create dataloaders
train_dataloader = dataset.get_dataloader('train')
test_dataloader = dataset.get_dataloader('test')
val_dataloader = dataset.get_dataloader('val')
print(train_dataloader)

# ---------------------------------
# Model Construction
# ---------------------------------

print('Constructing model.')
# hyperparams
num_hiddens, num_heads = 64, 4
batch_size, num_queries, num_kvpairs = 2, 4, 6

attention = T.MultiHeadAttention(num_hiddens, num_heads, 0.3, num_classes=len(class_names))

# if you have a past iteration of training...
#attention.load_state_dict(torch.load(f'model_weights_epochs29.pth'))

# ---------------------------------
# Optimizer, loss, embeddings
# ---------------------------------

# optimizer
optimizer = torch.optim.AdamW(attention.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1E-9)
class_counts = torch.bincount(torch.tensor(y, dtype=torch.long))

# loss
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()*len(class_weights)
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# embedding
embedding_dim = num_hiddens
input_projection = nn.Linear(1, embedding_dim)
pos_encoding = nn.Embedding(len(X[0,:]), embedding_dim)
## train for n epochs, then save model params
nn.init.xavier_uniform_(input_projection.weight)
nn.init.zeros_(input_projection.bias)

# ---------------------------------
# Training Loop
# ---------------------------------

epochs = 100
loss_train = []
loss_valid = []
accuracy_valid = []

attention = attention.to(device)
input_projection = input_projection.to(device)
pos_encoding = pos_encoding.to(device)

for i in range(epochs):
    print(f'Epoch {i+1}')

    #train
    attention.train()
    loss_vals = []
    for batch_idx, (batch_X, batch_y) in enumerate(train_dataloader):# load data from dataloader
         
        seq_len = batch_X.shape[1]
        batch_X = batch_X.unsqueeze(-1)
        embedded_X = input_projection(batch_X)
        positions = torch.arange(seq_len, device=device)
        embedded_X = embedded_X + pos_encoding(positions).unsqueeze(0)
    
        query = embedded_X 
        key = embedded_X
        value = embedded_X
        preds = attention.forward(queries=query, keys=key, values=value, valid_lens=None )
        
        if batch_idx == 0:
            # debug info- still useful in training
            print(f"batch_y shape: {batch_y.shape}")
            print(f"First 5 predictions: {preds.argmax(dim=1)[:5]}")
            print(f"average prediction: {torch.mean(preds.argmax(dim=1).float())}")
            print(f"First 5 labels: {batch_y[:5]}")
            print(f"First logits values:\n{preds[:5]}")
            print(f"Shape of logits:\n{preds.shape}")
            print(f"Num classes in batch_y: {len(torch.unique(batch_y))}")
             
        # compute loss
        loss = loss_fn(preds, batch_y)
        loss_vals.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del batch_X, batch_y, embedded_X, preds, loss
    loss_train.append(np.mean(loss_vals))
    del loss_vals
     # validate
    attention.eval()
    loss_vals = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(test_dataloader):
            batch_X = batch_X.unsqueeze(-1)
            seq_len = batch_X.shape[1]
            embedded_X = input_projection(batch_X)
            positions = torch.arange(seq_len, device=device)
            embedded_X = embedded_X + pos_encoding(positions).unsqueeze(0)
            embedded_X = input_projection(batch_X)
            query = embedded_X # (sequence_length, batch_size, embed_dim)
            key = embedded_X
            value = embedded_X
            preds = attention.forward(queries=query, keys=key, values=value, valid_lens=None )

            # Then compute loss
            loss = loss_fn(preds, batch_y)
            loss_vals.append(loss.item())

            # calculate accuracy
            correct += (preds.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

            del batch_X, batch_y, embedded_X, preds, loss

    loss_valid.append(np.mean(loss_vals))
    accuracy = correct / total
    accuracy_valid.append(accuracy)
    print(f'        accuracy: {accuracy:.4f}')
    print(f'        loss:     {loss_vals[-1]:.4f}')
    del loss_vals
    torch.cuda.empty_cache()

    loss_valid_np = np.array(loss_valid, dtype=float)
    loss_train_np = np.array(loss_train, dtype=float)
    accuracy_valid_np = np.array(accuracy_valid, dtype=float)
    np.savetxt(f'loss_valid_{i}.csv', loss_valid_np, delimiter=',')
    np.savetxt(f'loss_train_{i}.csv', loss_train_np, delimiter=',')
    np.savetxt(f'accuracy_valid_{i}.csv', accuracy_valid_np, delimiter=',')
 
    torch.save(attention.state_dict(), f'model_weights_epochs{i}.pth')
