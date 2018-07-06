import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Customized_Loss(nn.Module):

    def __init__(self, beta = 1000):
        super(Customized_Loss, self).__init__()
        self.beta = beta

    def forward(self, output, target):
        xys = target[:, :2]
        thetas = target[:, 2:]
        cosines = torch.cos(thetas)
        sines = torch.sin(thetas)


        xy_preds = output[:, :2]
        theta_preds = output[:, 2:]
        cosines_preds = torch.cos(theta_preds)
        sines_preds = torch.sin(theta_preds)

        xy_errors = torch.norm(xy_preds - xys, 2, dim = 1)
        cosines_errors = torch.norm(cosines_preds - cosines, 2, dim = 1)
        sines_errors = torch.norm(sines_preds - sines, 2, dim = 1)
        # theta0_errors = torch.norm(theta_preds - thetas, 2, dim = 1)
        # theta180_errors = torch.norm(np.pi - torch.abs(theta_preds - thetas), 2, dim = 1)
        # theta_errors_both = torch.stack((theta0_errors, theta180_errors), dim = 1)
        # theta_errors, _ = torch.min(theta_errors_both, dim = 1)

        # total_error = self.beta * torch.mean(xy_errors) + torch.mean(theta_errors)
        total_error = torch.mean(xy_errors) + self.beta * (torch.mean(cosines_errors) + torch.mean(sines_errors))
        return total_error
