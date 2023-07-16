import torch
import torch.nn as nn
from simple_UNet import UNet
import os
from hyperparameters import *
# from loss_function import DiceLoss

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = UNet(in_channels=3,
            out_channels=1,
            n_class=1,
            kernel_size=3,
            padding=1,
            stride=1).to(device)

n_iters = int(train_set_size / batch_size)
iterations = epochs * n_iters
step_size = 2*n_iters
if not os.path.exists('./results'):
    os.mkdir('./results')
save_PATH = f'./results/{epochs}epochs_{lr}lr_{batch_size}batch'
if not os.path.exists(save_PATH):
    os.mkdir(save_PATH)

model = model.float()
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr)

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs):

    train_loss, valid_loss, accuracy = [], [], []

    best_acc = 0.0

    for epoch in range(epochs):

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
                # x = torch.permute(x, (0, 3, 2, 1))
                # x = torch.permute(x, (0, 3, 1, 2))
                x = x.permute(0,3,1,2)

                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
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
                acc = acc_fn(outputs, y, batch_size)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
    
    return train_loss, valid_loss, accuracy

train_loss, valid_loss, accuracy = train(model, train_loader, valid_loader, loss_fn, opt, acc_fn, epochs)

#To save the model
PATH = save_PATH + '/network.pth'
torch.save(model.state_dict(), PATH)

# Save training measures
f = open(save_PATH + "/training_measures.csv", "w")
f.write("{},{},{}\n".format("Train Loss", "Valid Loss","Train Acc"))
for x in zip(train_loss, valid_loss, accuracy):
    f.write("{},{},{}\n".format(x[0], x[1], x[2]))
f.close()