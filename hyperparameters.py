from dataloader import DataLoaderSegmentation as DataLoaderSeg
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

torch.manual_seed(0)
batch_size = 5
lr = 0.001
epochs = 20

# Load data
dataset = DataLoaderSeg("dataset_2")

# Generate training, validation and test datasets
# random split
train_set_size = int(len(dataset)*0.6)
valid_set_size = int(len(dataset)*0.2)
test_set_size = len(dataset)-train_set_size-valid_set_size
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size, test_set_size])

# start to load from customized DataSet object
# bring batch size, iterations and epochs together
train_loader = DataLoader(train_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)
valid_loader = DataLoader(valid_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)
test_loader = DataLoader(test_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)

def out_to_mask(outputs_squeezed):
    # outpus shape [batch,512,512]
    # sigmoid =nn.Sigmoid()
    # mask = sigmoid(outpus_squeezed)
    outputs_squeezed[outputs_squeezed<=0.5] = 0
    outputs_squeezed[outputs_squeezed>0.5] = 1
    
    return outputs_squeezed
