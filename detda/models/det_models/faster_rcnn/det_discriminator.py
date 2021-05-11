import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class Det_Dis_inst(nn.Module):
    def __init__(self, fc_size=2048):
        super(Det_Dis_inst, self).__init__()
        self.fc_1_inst = nn.Linear(fc_size, 1024)
        self.fc_2_inst = nn.Linear(1024, 256)
        self.fc_3_inst = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax()
        # self.logsoftmax = nn.LogSoftmax()
        # self.bn = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.relu(self.fc_1_inst(x))
        x = self.relu((self.fc_2_inst(x)))
        x = self.relu(self.bn2(self.fc_3_inst(x)))
        return x


class Det_Dis_1(nn.Module):
    def __init__(self, channel_num=(256, 256, 128), context=False):
        super(Det_Dis_1, self).__init__()
        self.conv1 = nn.Conv2d(channel_num[0], channel_num[1], kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(channel_num[1], channel_num[2], kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(channel_num[2], 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x, use_amp=False):
        def _forward(x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            if self.context:
                feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
                x = self.conv3(x)
                return torch.sigmoid(x), feat
            else:
                x = self.conv3(x)
                return torch.sigmoid(x)

        if use_amp:
            with autocast():
                return _forward(x)
        else:
            return _forward(x)


class Det_Dis_2(nn.Module):
    def __init__(self, channel_num=(512, 512, 128, 128), context=False):
        super(Det_Dis_2, self).__init__()
        self.conv1 = conv3x3(channel_num[0], channel_num[1], stride=2)
        self.bn1 = nn.BatchNorm2d(channel_num[1])
        self.conv2 = conv3x3(channel_num[1], channel_num[2], stride=2)
        self.bn2 = nn.BatchNorm2d(channel_num[2])
        self.conv3 = conv3x3(channel_num[2], channel_num[3], stride=2)
        self.bn3 = nn.BatchNorm2d(channel_num[3])
        self.fc = nn.Linear(channel_num[3], 1)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_num = channel_num

    def forward(self, x, use_amp=False):
        def _forward(x):
            x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
            x = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = x.view(-1, self.channel_num[3])
            if self.context:
                feat = x
            x = self.fc(x)
            if self.context:
                return x, feat
            else:
                return x

        if use_amp:
            with autocast():
                return _forward(x)
        else:
            return _forward(x)


class Det_Dis_3(nn.Module):
    def __init__(self, channel_num=(512, 512, 128, 128), context=False):
        super(Det_Dis_3, self).__init__()
        self.conv1 = conv3x3(channel_num[0], channel_num[1], stride=2)
        self.bn1 = nn.BatchNorm2d(channel_num[1])
        self.conv2 = conv3x3(channel_num[1], channel_num[2], stride=2)
        self.bn2 = nn.BatchNorm2d(channel_num[2])
        self.conv3 = conv3x3(channel_num[2], channel_num[3], stride=2)
        self.bn3 = nn.BatchNorm2d(channel_num[3])
        self.fc = nn.Linear(channel_num[3], 1)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_num = channel_num

    def forward(self, x, use_amp=False):
        def _forward(x):
            x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
            x = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = x.view(-1, self.channel_num[3])
            if self.context:
                feat = x
            x = self.fc(x)
            if self.context:
                return x, feat
            else:
                return x

        if use_amp:
            with autocast():
                return _forward(x)
        else:
            return _forward(x)


#
class Det_Dis_2_Conv(nn.Module):
    def __init__(self, channel_num=(512, 512, 128, 128)):
        super(Det_Dis_2_Conv, self).__init__()
        self.conv1 = conv3x3(channel_num[0], channel_num[1], stride=1)
        self.bn1 = nn.BatchNorm2d(channel_num[1])
        self.conv2 = conv3x3(channel_num[1], channel_num[2], stride=1)
        self.bn2 = nn.BatchNorm2d(channel_num[2])
        self.conv3 = conv3x3(channel_num[2], channel_num[3], stride=1)
        self.bn3 = nn.BatchNorm2d(channel_num[3])
        self.conv4 = conv3x3(channel_num[3], 1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_num = channel_num

    def forward(self, x, use_amp=False):
        def _forward(x):
            x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
            x = self.conv4(x)
            return x

        if use_amp:
            with autocast():
                return _forward(x)
        else:
            return _forward(x)


class Det_Dis_3_Conv(nn.Module):
    def __init__(self, channel_num=(512, 512, 128, 128), ):
        super(Det_Dis_3_Conv, self).__init__()
        self.conv1 = conv3x3(channel_num[0], channel_num[1], stride=1)
        self.bn1 = nn.BatchNorm2d(channel_num[1])
        self.conv2 = conv3x3(channel_num[1], channel_num[2], stride=1)
        self.bn2 = nn.BatchNorm2d(channel_num[2])
        self.conv3 = conv3x3(channel_num[2], channel_num[3], stride=1)
        self.bn3 = nn.BatchNorm2d(channel_num[3])
        self.conv4 = conv3x3(channel_num[3], 1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_num = channel_num

    def forward(self, x, use_amp=False):
        def _forward(x):
            x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
            x = self.conv4(x)
            return x

        if use_amp:
            with autocast():
                return _forward(x)
        else:
            return _forward(x)