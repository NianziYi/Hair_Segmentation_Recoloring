from dataloader import DataLoaderSegmentation as DataLoaderSeg
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

torch.manual_seed(0)
batch_size = 20
lr = 0.001
epochs = 20

# Load data
#dataset_PATH = r"D:\Uni\SS23\Praktikum\Hair_Segmentation_Recoloring\dataset_2"
dataset_PATH = r"/Users/nianziyi/Desktop/Hair_Segmentation_Recoloring/dataset_2"
dataset = DataLoaderSeg(dataset_PATH)

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
    sigmoid =nn.Sigmoid()
    mask = sigmoid(outputs_squeezed)
    '''
    outputs_squeezed[outputs_squeezed<=0.5] = 0
    outputs_squeezed[outputs_squeezed>0.5] = 1
    '''
    mask[mask<=0.5] = 0
    mask[mask>0.5] = 1
    
    return mask

def acc_fn(outputs, y, batch_size):
    '''
    outputs: [batch, 1 , H, W]
    y: [batch, H, W]
    '''
    _, H, W = y.shape
    mask = out_to_mask(outputs)
    num_pixels = H * W
    
    acc = torch.zeros([batch_size, 1])
    
    for i in range(batch_size):
        equality_matrix = torch.eq(mask[i], y[i])
        num_corr_pred = equality_matrix.sum()
        acc_num = (num_corr_pred/num_pixels).item()
        acc[i] = acc_num
    acc_avg = acc.sum()/batch_size
    return acc_avg.item()  