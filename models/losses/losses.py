import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import norm
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
        win = ([9] * ndims if self.win is None else self.win)
        padding = math.floor(win[0] / 2)
        groups = Ii.shape[1]

        # compute filters
        sum_filt = torch.ones([groups, 1, *win], dtype = Ii.dtype, device = Ii.device)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, padding = padding, groups = groups)
        J_sum = conv_fn(Ji, sum_filt, padding = padding, groups = groups)
        I2_sum = conv_fn(I2, sum_filt, padding = padding, groups = groups)
        J2_sum = conv_fn(J2, sum_filt, padding = padding, groups = groups)
        IJ_sum = conv_fn(IJ, sum_filt, padding = padding, groups = groups)

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
        kernel = [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                  [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]
        self.register_buffer('kernel', torch.tensor(kernel)[:, None, None, ...])

    def forward(self, y_pred):
        grad = [F.conv2d(y_pred, self.kernel[i].expand(-1, y_pred.shape[1], -1, -1), padding = 1) for i in range(len(self.kernel))]
        grad = torch.mean(norm(torch.stack(grad, dim = -1), dim = -1, ord = self.penalty))

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

