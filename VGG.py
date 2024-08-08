import torch.nn as nn

class VGGConv(nn.Module):
  def __init__(self, in_channels:int, out_channels:int, ImgSize:int) -> None:
    super(VGGConv, self).__init__()
    self.block = nn.Sequential(
    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=(in_channels+3)//2),
    nn.ReLU(),
    nn.BatchNorm2d(out_channels)
    )

  def forward(self, x):
    y = self.block(x)
    return y
  
class VGG(nn.Module):
  def __init__(self, in_channels:int, ImgSize:int, numClasses:int) -> None:
    super(VGG, self).__init__()
    
    self.block1 = nn.Sequential(
    VGGConv(in_channels=in_channels, out_channels=64, ImgSize=224),
    VGGConv(in_channels=64, out_channels=64, ImgSize=224),
    nn.MaxPool2d(kernel_size=2, stride=2)
    ) 

    self.block2 = nn.Sequential(
    VGGConv(in_channels=64, out_channels=128, ImgSize=112),
    VGGConv(in_channels=128, out_channels=128, ImgSize=112),
    nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.block3 = nn.Sequential(
    VGGConv(in_channels=128, out_channels=256, ImgSize=56),
    VGGConv(in_channels=256, out_channels=256, ImgSize=56),
    VGGConv(in_channels=256, out_channels=256, ImgSize=56),
    nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    self.block4 = nn.Sequential(
    VGGConv(in_channels=256, out_channels=512, ImgSize=28),
    VGGConv(in_channels=512, out_channels=512, ImgSize=28),
    VGGConv(in_channels=512, out_channels=512, ImgSize=28),
    nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    self.block5 = nn.Sequential(
    VGGConv(in_channels=512, out_channels=512, ImgSize=14),
    VGGConv(in_channels=512, out_channels=512, ImgSize=14),
    VGGConv(in_channels=512, out_channels=512, ImgSize=14),
    nn.MaxPool2d(kernel_size=2, stride=2) # 7*7*512
    ) 

    self.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(7*7*512, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1000),
    nn.ReLU(),
    nn.Linear(1000, numClasses)
    )

  def forward(self, x):
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3)
    x5 = self.block5(x4)
    y = self.fc(x5)
    return y
  
if __name__ == '__main__':
  model = VGG(in_channels=3, ImgSize=224, numClasses=2).cuda()