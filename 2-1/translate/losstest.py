import torch
import torch.nn as nn
import torch.nn.functional as F

softmax = nn.LogSoftmax(dim = 1)
s1= nn.Softmax(dim=1)
 
 
torch.manual_seed(2019)
output = torch.randn(2, 3)  # 网络输出
print(s1(output))
output = softmax(output)
target = torch.ones(2, dtype=torch.long).random_(3)  # 真实标签
print(output)
print(target)
 
# 直接调用
loss = F.nll_loss(output, target)
print(loss)
 
# 实例化类
criterion = nn.NLLLoss(reduction='none')
loss = criterion(output, target)
print(loss)