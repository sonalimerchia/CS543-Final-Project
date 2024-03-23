import torch.nn as nn
import torchvision.models as tvm

class CnnEncoder(nn.Module):
    def __init__(self, C):
        # pool5
        super(CnnEncoder, self).__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )

        self.s2 = nn.Sequential(
            # input channels, output channels, kernel size
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.s3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        s1_out = self.s1(x)
        s2_out = self.s2(s1_out)
        s3_out = self.s3(s2_out)

        return s1_out, s2_out, s3_out