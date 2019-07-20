import torch.nn as nn

class ConvNet_single(nn.Module):
    def __init__(self):
        super(ConvNet_single, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.l1_batchnorm=nn.BatchNorm2d(64)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.l2_batchnorm=nn.BatchNorm2d(128)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.l3_batchnorm=nn.BatchNorm2d(256)
        
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.l4_batchnorm=nn.BatchNorm2d(512)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        
        self.l5_batchnorm=nn.BatchNorm2d(1024)
        
        self.leakyrelu=nn.LeakyReLU()
        
        self.fc = nn.Linear(2*2*1024,1000)
        self.fc01= nn.Linear(1000,500)
        self.fc1= nn.Linear(500,9)

    def forward(self, x):
        out = self.layer1(x)
        out = self.l1_batchnorm(out)
        out = self.layer2(out)
        out = self.l2_batchnorm(out)
        out=  self.layer3(out)
        out = self.l3_batchnorm(out)
        out=  self.layer4(out)
        out = self.l4_batchnorm(out)
        out=  self.layer5(out)
        out = self.l5_batchnorm(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out=self.leakyrelu(out)
        out= self.fc01(out)
        out=self.leakyrelu(out)
        out= self.fc1(out)
        return out
