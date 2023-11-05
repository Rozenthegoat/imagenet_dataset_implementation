import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data_loader import *
from helper_func import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet()
model.to(device)
print(model)

# hyper parameter
epochs = 5
lr = 1e-2
batch_size = 4

# define loss function (criterion), optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=0.9,
                            weight_decay=0.0005)

"""Sets the learning rate ot the initial LR decayed by 10 every 30 epochs"""
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


train_loader, val_loader = data_loader(root='./imagenet/', batch_size=batch_size)
print(train_loader)
print(val_loader)

#
# end = time.time()
# for i, (images, target) in enumerate(train_loader):
#     # measure data loading time
#     data_time.update(time.time() - end)
#
#     # move data to the same device as model
#     images = images.to(device, non_blocking=True)
#     target = target.to(device, non_blocking=True)
#
#     # compute output
#     output = model(images)
#     loss = criterion(output, target)
#
#     # measure accuracy and record loss
#     acc1, acc5 = accuracy(output, target, topk=(1, 5))
#     losses.update(loss.item(), images.size(0))
#     top1.update(acc1[0], images.size(0))
#     top5.update(acc5[0], images.size(0))
#
#     # compute gradient and do SGD step
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # measure elapsed time
#     batch_time.update(time.time() - end)
#     end = time.time()
#
#     """In Pytorch's official guidance, here is if % args.print_freq == 0"""
#     if i % 10 == 0:
#         progress.display(i + 1)


for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if torch.cuda.is_available():
        #     data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader)}%)]\tLoss: {loss.item()}')
            # print(f"max_memory_allocated: {torch.cuda.max_memory_allocated()/1000000}")

    for batch_idx, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
