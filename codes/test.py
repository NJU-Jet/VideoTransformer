import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

class My(nn.Module):
    def __init__(self, a):
        super(My, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        a.append(2)
    def forward(self, x):
        out = self.conv(x)
        return out


b = [1, 2]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
test = My(b).to(device)
print(b)
#test = nn.DataParallel(test)
#print(test.module.state_dict().keys())
#print(sum([x.numel() for x in test.parameters()]))
#for m in test.modules():
#    print(type(m.modules()))
