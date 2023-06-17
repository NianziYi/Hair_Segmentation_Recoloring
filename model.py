import torch
import torch.nn as nn

torch.manual_seed(0)

# Implement UNET
# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.RELU()
        
    def forward(self, inputs):
            x = self.conv1(inputs)
            # x = self.bn1(x)
            x = self.relu(x)
            
            x = self.conv2(x)
            # x = self.bn2(x)
            x = self.relu(x)
            return x

# Encoder Block
class encoder_block(nn.Module):
    '''
    Output:
    x: output of conv_block, input of pooling layer
    p: output of pooling layer
    '''
    def __init__(self, in_c, out_c):
        super.__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
# Decoder Block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x    
    
# UNET Architecture
class unet(nn.Module):
    def __init__(self):
        super().__init()
        # Encoder
        self.e1 = encoder_block(3,64)
        self.e2 = encoder_block(64,128)
        self.e3 = encoder_block(128,256)
        self.e4 = encoder_block(256,512)
        
        # Bottleneck
        self.b = conv_block(512,1024)
        
        # Decoder
        self.d1 = decoder_block(1024,512)
        self.d2 = decoder_block(512,256)
        self.d3 = decoder_block(256,128)
        self.d4 = decoder_block(128,64)
        
        # Classifier
        self.outputs = nn.Conv(64,1, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e1(p1)
        s3, p3 = self.e1(p2)
        s4, p4 = self.e1(p3)
        
        # Bottleneck
        b = self.b(p4)
        
        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d1(d1, s3)
        d3 = self.d1(d2, s2)
        d4 = self.d1(d3, s1)
        
        # Classifier
        outputs = self.outputs(64)
        
        return outputs