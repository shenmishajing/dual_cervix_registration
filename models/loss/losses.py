import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win = None):
        super().__init__()
        self.win = win

    def forward(self, y_pred, y_true):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [Ii.shape[1]] + ([9] * ndims if self.win is None else self.win)

        # compute filters
        sum_filt = torch.ones([1, *win], dtype = Ii.dtype, device = Ii.device)

        pad_no = math.floor(win[0] / 2)

        stride = (1,) * ndims
        padding = (pad_no,) * ndims

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride = stride, padding = padding)
        J_sum = conv_fn(Ji, sum_filt, stride = stride, padding = padding)
        I2_sum = conv_fn(I2, sum_filt, stride = stride, padding = padding)
        J2_sum = conv_fn(J2, sum_filt, stride = stride, padding = padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride = stride, padding = padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum / win_size - u_I * u_J
        I_var = I2_sum / win_size - u_I * u_I
        J_var = J2_sum / win_size - u_J * u_J

        cc = cross / torch.sqrt(torch.clamp(I_var * J_var, min = 1e-7))

        return 1 - torch.mean(cc)


class MSE(nn.Module):
    """
    Mean squared error loss.
    """

    def forward(self, y_pred, y_true):
        return torch.mean((y_true - y_pred) ** 2)


class Dice(nn.Module):
    """
    N-D dice for segmentation
    """

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim = vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim = vol_axes), min = 1e-7)
        dice = torch.mean(top / bottom)
        return 1 - dice


class Grad(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty = 1, loss_mult = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        self.register_buffer('kernel_x', torch.tensor(kernel_x))
        self.register_buffer('kernel_y', torch.tensor(kernel_y))

    def forward(self, y_pred):
        dx = torch.pow(torch.abs(F.conv2d(y_pred, self.kernel_x[None, None, ...].repeat(1, y_pred.shape[1], 1, 1), padding = 1)),
                       self.penalty)
        dy = torch.pow(torch.abs(F.conv2d(y_pred, self.kernel_y[None, None, ...].repeat(1, y_pred.shape[1], 1, 1), padding = 1)),
                       self.penalty)
        grad = torch.mean(torch.pow(torch.clamp(dx + dy, min = 1e-7), 1 / self.penalty))

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
