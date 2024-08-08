import torch

class AlexNet(torch.nn.Module):
  def __init__(self, classNum = 2):
    self.classNum = classNum
    super(AlexNet, self).__init__()
    # 输入维度为 224 * 224 *3
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
    self.reLu1 = torch.nn.ReLU()
    self.maxPool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
    self.reLu2 = torch.nn.ReLU()
    self.maxPool2 = torch.nn.MaxPool2d(kernel_size=3,stride=2)

    self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
    self.reLu3 = torch.nn.ReLU()
    self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
    self.reLu4 = torch.nn.ReLU()
    self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
    self.reLu5 = torch.nn.ReLU()
    self.maxPool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    self.flatten = torch.nn.Flatten(start_dim=0)
    self.linear1 = torch.nn.Linear(6400, 4096)
    self.reLu6 = torch.nn.ReLU()
    self.dropout1 = torch.nn.Dropout()

    self.linear2 = torch.nn.Linear(4096, 4096)
    self.reLu7 = torch.nn.ReLU()
    self.dropout2 = torch.nn.Dropout()

    self.linear3 = torch.nn.Linear(4096, self.classNum)



  def forward(self, x):
    x = self.conv1(x)
    x = self.reLu1(x)
    x = self.maxPool1(x)

    x = self.conv2(x)
    x = self.reLu2(x)
    x = self.maxPool2(x)

    x = self.conv3(x)
    x = self.reLu3(x)
    x = self.conv4(x)
    x = self.reLu4(x)
    x = self.conv5(x)
    x = self.reLu5(x)
    x = self.maxPool3(x)

    x = self.flatten(x)
    print("flatten", x)
    x = self.linear1(x)
    print("linear1", x)
    x = self.reLu6(x)
    x = self.dropout1(x)

    x = self.linear2(x)
    x = self.reLu7(x)
    x = self.dropout2(x)

    x = self.linear3(x)
    return x