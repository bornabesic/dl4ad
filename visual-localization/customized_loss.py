import torch
import torch.nn as nn
import torch.nn.functional as F

class Customized_Loss(nn.Module):

    def __init__(self, beta = 1000):
        super(Customized_Loss, self).__init__()
        self.beta = beta

    def forward(self, output, target):
        Pos = target[:, :2]
        #Hardcoded re-normalization
        Pos[0] = Pos[0] * 350
        Pos[1] = Pos[1] * 350
        Quat = target[:, 2:]
        Pos_est = output[:,:2]
        Pos_est[0] = Pos_est[0] * 350
        Pos_est[1] = Pos_est[1] * 350
        Quat_est = output[:,2:]
        Quat_est_normalized = F.normalize(Quat_est)
        Pos_error = torch.norm(Pos_est - Pos, 2, dim = 1)
        Quat_error = torch.norm(Quat_est_normalized - Quat, dim = 1)
        total_error = torch.mean(Pos_error) + self.beta * torch.mean(Quat_error)
        return total_error
