import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

__all__ = ['WideResNet']

"""
Adapted from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_layout, stride, drop_out=0.0, bias=False):
        super(BasicBlock, self).__init__()
        stride_used = False if stride==2 else True
        self.kernel_layout = kernel_layout
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if not stride_used and kernel_layout[0] == 3:
            s = stride
            stride_used = True
        else:
            s = 1
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_layout[0], stride=s,
                                   padding=1 if kernel_layout[0] == 3 else 0, bias=bias)
        
        if len(kernel_layout) == 3:
            if not stride_used and kernel_layout[1] == 3:
                s = stride
                stride_used = True
            else:
                s = 1
            self.seq = nn.Sequential(
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_layout[1], stride=s,
                                       padding=1 if kernel_layout[1] == 3 else 0, bias=bias),
            )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        if not stride_used and kernel_layout[-1] == 3:
            s = stride
            stride_used = True
        else:
            s = 1
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_layout[-1], stride=s,
                               padding=1 if kernel_layout[-1] == 3 else 0, bias=bias)
        self.drop_out = drop_out
        self.equalInOut = in_planes == out_planes
        self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=bias) if not self.equalInOut else None

        assert stride_used, "stride was not used"
    
    def forward(self, x):

        if self.equalInOut:
            out = self.relu1(self.bn1(x))
        else:
            x = self.relu1(self.bn1(x))
        # first conv
        tmp = out if self.equalInOut else x
        out = self.conv1(tmp)

        # mid conv
        if len(self.kernel_layout) == 3:
            out = self.seq(out)
        
        # last conv
        out = self.relu2(self.bn2(out))
        if self.drop_out > 0:
            out = F.dropout(out, p=self.drop_out, training=self.training)
        out = self.conv2(out)
        #print("out.shape: ", out.shape, "x.shape: ", x.shape)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
class BasicBlock_vary_l(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_layout, stride, drop_out=0.0, bias=False):
        super(BasicBlock_vary_l, self).__init__()
        assert drop_out == 0.0, "dropout not supported"
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_layout[0], stride=stride,
                               padding=1, bias=bias)
        
        self.layer = []
        for i in range(1, len(kernel_layout)):
            self.layer.append(nn.BatchNorm2d(out_planes))
            self.layer.append(nn.ReLU(inplace=True))
            self.layer.append(nn.Conv2d(out_planes, out_planes, kernel_size=kernel_layout[i], stride=1,
                                       padding=1, bias=bias))
        self.layer = nn.Sequential(*self.layer)

        self.equalInOut = in_planes == out_planes
        self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=bias) if not self.equalInOut else None
        
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
        else:
            x = self.relu1(self.bn1(x))
        # first conv
        tmp = out if self.equalInOut else x
        out = self.conv1(tmp)

        if len(self.layer)>0:
            out = self.layer(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, kernel_layout, block, stride, drop_out=0.0, bias=False):
        super(NetworkBlock, self).__init__()
        self.layer = []

        if len(kernel_layout) == 3 and not kernel_layout == [3,3,3] or len(kernel_layout) == 2:
            for i in range(int(nb_layers)):
                stride = stride if i == 0 else 1
                in_planes = in_planes if i == 0 else out_planes
                self.layer.append(block(in_planes, out_planes, kernel_layout, stride, drop_out, bias=bias))
        elif len(kernel_layout) in [1,3,4]:
            for i in range(int(nb_layers)):
                stride = stride if i == 0 else 1
                in_planes = in_planes if i == 0 else out_planes
                self.layer.append(BasicBlock_vary_l(in_planes, out_planes, kernel_layout, stride, drop_out, bias=bias))
        else:
            raise ValueError("kernel_layout must be of length 2, 3 or 4")
        
        self.layer = nn.Sequential(*self.layer)
       
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, layout, kernel_size, kernel_layout, padding, input_channels, \
                 num_classes, widen_factor=1, bias=False, drop_out=0.0, restrict=None):
        super(WideResNet, self).__init__()
        # nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels = [layout[0], layout[1]*widen_factor, layout[2]*widen_factor, layout[3]*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        if len(kernel_layout) == 1:
            n = int(n * 2)
        elif len(kernel_layout) == 4:
            n = int(n / 2)
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(input_channels, nChannels[0], kernel_size=kernel_size, stride=1,
                               padding=1, bias=bias)
        # 1st block
        self.layer1 = NetworkBlock(n, nChannels[0], nChannels[1], kernel_layout, block, 1, drop_out, bias=bias)
        # 2nd block
        self.layer2 = NetworkBlock(n, nChannels[1], nChannels[2], kernel_layout, block, 2, drop_out, bias=bias)
        # 3rd block
        self.layer3 = NetworkBlock(n, nChannels[2], nChannels[3], kernel_layout, block, 2, drop_out, bias=bias)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(f"Kernel layout: ", cfg.model.kernel_layout)
    inp = torch.rand(1, 1, 32, 32)
    n_inputs = inp.shape[1]
    n_outputs = 10
    # depth, num_classes, widen_factor=1, dropRate=0.0
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
        )
    # tot_param = sum([p.numel() for p in net.conv1.parameters()  if p.requires_grad])
    tot_param = sum([p.numel() for p in net.parameters()  if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param)) # total 2.748.890 # block1 121248
    #print(net.layer1)

    inp = inp# .cuda()
    net# .cuda()
    print(net(inp).size())

    #y = net(torch.randn(1,3,32,32))
    #print(y.size())


if __name__ == "__main__":
    main()
