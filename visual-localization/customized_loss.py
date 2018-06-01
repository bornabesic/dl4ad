import warnings

import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F
from torch.autograd.variable import Variable



def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"



class Customized_Loss(Module):
    def __init__(self, beta, size_average=True, reduce=True):
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        Pos_est = Variable(input(:,:3), requires_grad=True)
        Quat_est = Variable(input(:,3:),requires_grad=True)
        Quat_est_amount = torch.mm(Quat_est,torch.t(Quat_est))
        Quat_div = torch.div(Quat_est,Quat_est_amount)
        return F.mse_loss(Pos_est, target(:,:3), size_average=self.size_average, reduce=self.reduce) + self.beta  * F.mse_loss(Quat_div, target(:,3:), size_average=self.size_average, reduce=self.reduce)


loss = Customized_Loss(beta = 1000)
input = torch.randn(3, 6, requires_grad=True)
target = torch.randn(3, 6)
output = loss(input, target)
output.backward()
  