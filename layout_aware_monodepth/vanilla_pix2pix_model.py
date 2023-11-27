import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthModel(nn.Module):
    # initializers
    def __init__(self, d=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        self.upconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.upconv1_bn = nn.BatchNorm2d(d * 8)
        self.upconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.upconv2_bn = nn.BatchNorm2d(d * 8)
        self.upconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.upconv3_bn = nn.BatchNorm2d(d * 8)
        self.upconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.upconv4_bn = nn.BatchNorm2d(d * 8)
        self.upconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.upconv5_bn = nn.BatchNorm2d(d * 4)
        self.upconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.upconv6_bn = nn.BatchNorm2d(d * 2)
        self.upconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.upconv7_bn = nn.BatchNorm2d(d)
        self.upconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

        self.final_conv = nn.Conv2d(3, 1, 3, 1, 1)

    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        d1 = F.dropout(self.upconv1_bn(self.upconv1(F.relu(e8))), 0.5, training=True)
        d1 = pad_to_match(e7, d1)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.upconv2_bn(self.upconv2(F.relu(d1))), 0.5, training=True)
        d2 = pad_to_match(e6, d2)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.upconv3_bn(self.upconv3(F.relu(d2))), 0.5, training=True)
        d3 = pad_to_match(e5, d3)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.upconv4_bn(self.upconv4(F.relu(d3)))
        d4 = pad_to_match(e4, d4)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.upconv5_bn(self.upconv5(F.relu(d4)))
        d5 = pad_to_match(e3, d5)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.upconv6_bn(self.upconv6(F.relu(d5)))
        d6 = pad_to_match(e2, d6)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.upconv7_bn(self.upconv7(F.relu(d6)))
        d7 = pad_to_match(e1, d7)
        d7 = torch.cat([d7, e1], 1)
        d8 = self.upconv8(F.relu(d7))

        d9 = self.final_conv(F.relu(d8))
        o = F.sigmoid(d9)

        return o


def pad_to_match(x, y):
    diffY = x.size()[2] - y.size()[2]
    diffX = x.size()[3] - y.size()[3]
    y = F.pad(y, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    return y


if __name__ == "__main__":
    model = DepthModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
