import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_shape, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape)
        if in_channels == out_channels and stride_shape == 1:
            self.input_isConv = False
        else:
            self.input_isConv = True

        self.batch_norm_ip = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = self.seq(self.input_tensor)
        if self.input_isConv:
            self.input_tensor = self.conv1X1(self.input_tensor)
        self.input_tensor = self.batch_norm_ip(self.input_tensor)
        output_tensor += self.input_tensor
        output_tensor = self.relu2(output_tensor)
        return output_tensor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(batch_dim, -1)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.seq = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Dropout(p=0.1),
            ResBlock(in_channels=64, out_channels=64),
            #nn.Dropout(p=0.1),
            ResBlock(in_channels=64, out_channels=128, stride_shape=2),
            nn.Dropout(p=0.5),
            ResBlock(in_channels=128, out_channels=256, stride_shape=2),
            nn.Dropout(p=0.5),
            ResBlock(in_channels=256, out_channels=512, stride_shape=2),
            nn.AvgPool2d(kernel_size=10),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        output_tensor = self.seq(input_tensor)
        return output_tensor

