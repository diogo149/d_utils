import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_p=0.5):
        super(MLP, self).__init__()
        all_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = []
        for idx, (s1, s2) in enumerate(zip(all_sizes, all_sizes[1:])):
            if idx != 0:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout_p))
            self.layers.append(nn.Linear(s1, s2))

        # HACK to register the modules
        # i.e. to get their parameters
        self.sequential = nn.Sequential(*self.layers)


    def forward(self, x):
        tmp = torch.flatten(x, 1)
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # TODO should this log softmax be removed?
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        C = num_channels
        self.conv1 = nn.Conv2d(in_channels=C,
                               out_channels=C * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=C * 8,
                               out_channels=C * 32,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=C * 32,
                               out_channels=C * 128,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(C * 128, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
