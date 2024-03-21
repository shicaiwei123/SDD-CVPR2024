import torch.nn as nn
import torch


class SPP(nn.Module):
    def __init__(self, isaverage=True):
        super(SPP, self).__init__()
        if isaverage:
            self.normal_pooling = nn.AdaptiveAvgPool2d((4, 4))
            self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.normal_pooling = nn.AdaptiveMaxPool2d((4, 4))
            self.pooling_2x2 = nn.AdaptiveMaxPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x_normal = self.normal_pooling(x)
        x_2x2 = self.pooling_2x2(x_normal)
        x_1x1 = self.pooling_1x1(x_normal)

        x_normal_flatten = torch.flatten(x_normal, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        x_feature = torch.cat((x_normal_flatten, x_2x2_flatten, x_1x1_flatten), dim=2)
        x_strength = x_feature.permute((2, 0, 1))
        x_strength = torch.mean(x_strength, dim=2)

        # print(x_feature.shape)

        # normal
        # x_feature_norm = torch.sqrt(torch.sum(x_feature ** 2, dim=1, keepdim=True))
        # x_feature = x_feature / (x_feature_norm + 1e-6)
        # x_feature[x_feature != x_feature] = 0

        return x_feature, x_strength
