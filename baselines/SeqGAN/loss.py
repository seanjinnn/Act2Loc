'''
policy gradient loss function
'''
import torch
import torch.nn as nn


class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward()) 
    for adversarial training of Generator
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward):
        """
        Inputs: pred, target, reward
            - pred: (batch_size, seq_len), 
            - target : (batch_size, seq_len), 
            - reward : (batch_size, ), reward of each whole sentence
        """
        one_hot = torch.zeros(pred.size(), dtype=torch.uint8)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        # scatter_(input, dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input。可以理解成放置元素或者修改元素
        one_hot.scatter_(1, target.data.view(-1, 1), 1)     # 将target中的数据按照dim=1的方向填充到one_hot中
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward.contiguous().view(-1)
        loss = -torch.sum(loss)
        return loss
