from torch import nn


def initialize_weight(m):
    '''
    初始化权重
    w采用xavier_uniform_初始化
    b采用0初始化
    '''
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
