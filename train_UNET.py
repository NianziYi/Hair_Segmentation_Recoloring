import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloader import dataset
from unet import UNet
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

batch_size = 5
epochs = 10
lr = 0.001

# Generate training, validation and test datasets
# random split
train_set_size = int(len(dataset)*0.6)
valid_set_size = int(len(dataset)*0.2)
test_set_size = len(dataset)-train_set_size-valid_set_size
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size, test_set_size])

# start to load from customized DataSet object
# bring batch size, iterations and epochs together
batch_size = 5
train_loader = DataLoader(train_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)
valid_loader = DataLoader(valid_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)
test_loader = DataLoader(test_set,batch_size,shuffle=False,drop_last=True,pin_memory=False)


model = UNet(in_channels=3,
            out_channels=1,
            n_blocks=4,
            start_filters=32,
            activation='relu',
            normalization='batch',
            conv_mode='same',
            dim=2)
x = torch.randn(size=(3, 3, 1024,1024), dtype=torch.float32)
with torch.no_grad():
    out = model(x)

n_iters = int(train_set_size / batch_size)
iterations = epochs * n_iters
step_size = 2*n_iters
model_name ="{}epochs_lr{}_step{}".format(epochs, lr, step_size)
save_PATH = './model_name'
if not os.path.exists(save_PATH):
    os.mkdir(save_PATH)


model = model.float()
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr)


def out_to_mask(outpus_squeezed):
    #outpus shape [batch,512,512]
    sigmoid =nn.Sigmoid()
    mask = sigmoid(outpus_squeezed)
    mask[mask<=0.5] = 0
    mask[mask>0.5] = 1
    
    return mask


def acc_fn(outputs, y):
    mask = out_to_mask(outputs)
    num_samples = len(y)
    equality_matrix = torch.eq(mask, y)
    num_corr_pred = equality_matrix.sum()
    acc = num_corr_pred/num_samples
    return acc   


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
    
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = torch.permute(x, (0, 3, 2, 1))
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x) # outputs: [batch,1,512,512]
                    outputs = torch.squeeze(outputs)
                    y = y.to(torch.float64)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        outputs = torch.squeeze(outputs)
                        loss = loss_fn(outputs, y)

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
        
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   
            
    return train_loss, valid_loss

#To save the model
PATH = 'network.pth'
torch.save(model.state_dict(), PATH)