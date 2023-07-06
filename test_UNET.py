from train_UNET import test_loader
import torch
import unet

PATH = '/trained_model/network.pth'

unet = UNet()
unet.load_state_dict(torch.load(PATH))

dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = unet(images)
imshow(outputs[0])