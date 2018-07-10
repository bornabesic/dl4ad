import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Customized_Loss(nn.Module):

    def __init__(self, beta = 1000):
        super(Customized_Loss, self).__init__()
        self.beta = beta

    def forward(self, output, target):
        # Ground truth
        xys = target[:, :2]
        cos_sin = target[:, 2:4]

        # Predictions
        xy_preds = output[:, :2]
        cos_sin_preds = output[:, 2:4]
        cos_sin_preds = F.normalize(cos_sin_preds)

        # Position error
        xy_diff = xy_preds - xys
        xy_errors = torch.norm(xy_diff, p = 2, dim = 1)

        # Orientation error
        cos_sin_diff = cos_sin_preds - cos_sin
        cos_sin_errors = torch.norm(cos_sin_diff, p = 2, dim = 1)

        ''' Assertions '''
        inf = float("inf")

        assert not torch.isnan(xy_errors).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (xy_errors == inf).byte().any() # All elements in the tensor are zero (no infs)

        assert not torch.isnan(cos_sin_errors).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (cos_sin_errors == inf).byte().any() # All elements in the tensor are zero (no infs)

        ''''''

        # theta0_errors = torch.norm(theta_preds - thetas, 2, dim = 1)
        # theta180_errors = torch.norm(np.pi - torch.abs(theta_preds - thetas), 2, dim = 1)
        # theta_errors_both = torch.stack((theta0_errors, theta180_errors), dim = 1)
        # theta_errors, _ = torch.min(theta_errors_both, dim = 1)

        # total_error = self.beta * torch.mean(xy_errors) + torch.mean(theta_errors)
        angle_error = self.beta * torch.mean(cos_sin_errors)
        position_error = torch.mean(xy_errors) 
        # print(position_error, angle_error)
        total_error = position_error + angle_error

        ''' Assertions '''
        assert not torch.isnan(total_error).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (total_error == inf).byte().any() # All elements in the tensor are zero (no infs)
        ''''''

        # print(torch.mean(xy_errors), torch.mean(cosines_errors), torch.mean(sines_errors), sep = "\n")

        return total_error
