import torch.nn as nn
import torch


class SPP(nn.Module):
    def __init__(self, M=None):
        super(SPP, self).__init__()
        self.pooling_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

        self.M = M
        print(self.M)

    def forward(self, x):
        x_4x4 = self.pooling_4x4(x)
        x_2x2 = self.pooling_2x2(x_4x4)
        x_1x1 = self.pooling_1x1(x_4x4)

        x_4x4_flatten = torch.flatten(x_4x4, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        if self.M == '[1,2,4]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten, x_4x4_flatten), dim=2)
        elif self.M == '[1,2]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten), dim=2)
        else:
            raise NotImplementedError('ERROR M')

        x_strength = x_feature.permute((2, 0, 1))
        x_strength = torch.mean(x_strength, dim=2)


        return x_feature, x_strength
