import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.special import expit

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size = 11, stride = 4, padding =1),
            nn.ReLU(), nn.MaxPool2d(kernel_size = 3, stride =2),
            nn.LazyConv2d(256, kernel_size = 5, padding = 2),nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride =2),
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.LazyConv2d(256, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride =2), nn.Flatten(),
            nn.LazyLinear(4096),nn.ReLU(),
            nn.LazyLinear(4096),nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return logits
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, grayscale=False):
        super().__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = torch.nn.Sequential(
            
            torch.nn.Conv2d(in_channels, 6, kernel_size=3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16*8*8, 100*in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(100*in_channels, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits
    
class VGG6(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG6, self).__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [1,1,1,1],num_classes=num_classes)


'''GoogLeNet with PyTorch.'''

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class LogisticNet(nn.Module):
    def __init__(self, input_dim = 50):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear(x)
        #x = self.sigmoid(x)
        return x.squeeze()

class CDFNet(nn.Module):
    """Solve conditional ODE. Single output dim."""
    def __init__(self, D, hidden_dim=32, output_dim=1, device="cpu",
                 nonlinearity=nn.Tanh, n=10, lr=1e-3):
        super().__init__()
        
        self.output_dim = output_dim

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"CondODENet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"CondODENet: {device} specified, {self.device} used")

        self.first_layer = nn.Linear(D, 1)

        self.dudt = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.subNet = LogisticNet()
        
    def mapping(self, t):
        t = t[:, None]
        a = -10
        b = t
        tau = torch.matmul((b - a)/2, self.u_n) + (b+a)/2 # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = (b-a)/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1))
        #return torch.tanh(pred).squeeze()
        return pred.squeeze()
    
    # def mapping_prime(self, t):
    #     t = t[:,None].to(self.device)
    #     tau = torch.matmul(t/2, 1+self.u_n) # N x n
    #     tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
    #     f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
    #     tau2_ = torch.tensor(tau_, requires_grad=True)
    #     f_n2 = self.dudt(tau2_).reshape((*tau.shape, self.output_dim))
       
    #     dfn_dx = torch.autograd.grad(f_n2, tau2_, grad_outputs=torch.ones_like(f_n2))[0]
    #     pred_prime = 1/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1)) + t/2 * ((self.w_n[:,:,None] * dfn_dx).sum(dim=1)) + 0.5*(self.u_n + 1)
    #     return pred_prime
    

    def forward(self, t):
        F = self.mapping(t)
        du = self.dudt(t[:,None]).squeeze()
        #return -(torch.log(du) + torch.log(1-F**2))
        return -(torch.log(du) + torch.log(torch.tensor([2])) + F - 2*torch.log(1+torch.exp(F)))
        #return -(torch.log(du) + F - 2*torch.log(1+torch.exp(F)))
    
    def sum_forward(self, t):
        return self.forward(t).sum()
    
    def optimise(self, t, niters):
        print("X shape:", t.shape)
        for i in range(niters):
            self.optimizer.zero_grad()
            t2 = self.first_layer(t)
            t2 = t2.flatten()
            loss = self.sum_forward(t2)
            loss.backward()
            self.optimizer.step()
            if i % 100 == 0:
                print(loss.item())

    def mapping_logistic(self, x, p):
        t = p - self.subNet(x)
        t = t[:,None].to(self.device)
        tau = torch.matmul(t/2, 1+self.u_n) # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = t/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1))
        pred = torch.tanh(pred).squeeze()
        pred[pred<0] = 0
        return pred # F_0(x)
