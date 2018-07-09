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
        cosines = target[:, 2]
        sines = target[:, 3]

        # Predictions
        xy_preds = output[:, :2]
        cosines_preds = output[:, 2]
        sines_preds = output[:, 3]

        # Position error
        xy_diff = xy_preds - xys
        xy_errors = torch.norm(xy_diff, p = 2, dim = 1)

        # Orientation error
        cosines_diff = cosines_preds - cosines
        cosines_errors = torch.norm(cosines_diff, p = 2)

        sines_diff = sines_preds - sines
        sines_errors = torch.norm(sines_diff, p = 2)

        ''' Assertions '''
        inf = float("inf")

        assert not torch.isnan(xy_errors).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (xy_errors == inf).byte().any() # All elements in the tensor are zero (no infs)

        assert not torch.isnan(cosines_errors).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (cosines_errors == inf).byte().any() # All elements in the tensor are zero (no infs)

        assert not torch.isnan(sines_errors).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (sines_errors == inf).byte().any() # All elements in the tensor are zero (no infs)
        ''''''

        # theta0_errors = torch.norm(theta_preds - thetas, 2, dim = 1)
        # theta180_errors = torch.norm(np.pi - torch.abs(theta_preds - thetas), 2, dim = 1)
        # theta_errors_both = torch.stack((theta0_errors, theta180_errors), dim = 1)
        # theta_errors, _ = torch.min(theta_errors_both, dim = 1)

        # total_error = self.beta * torch.mean(xy_errors) + torch.mean(theta_errors)
        angle_error = self.beta * (torch.mean(cosines_errors) + torch.mean(sines_errors))
        position_error = torch.mean(xy_errors) 
        # print(position_error, angle_error)
        total_error = position_error + angle_error

        ''' Assertions '''
        assert not torch.isnan(total_error).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (total_error == inf).byte().any() # All elements in the tensor are zero (no infs)
        ''''''

        # print(torch.mean(xy_errors), torch.mean(cosines_errors), torch.mean(sines_errors), sep = "\n")

        return total_error
