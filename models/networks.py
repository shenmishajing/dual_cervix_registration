import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.normal import Normal
from pytorch_lightning import LightningModule
from . import layers, losses

from typing import Sequence, Optional, Union, Mapping, Any


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride = 1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape = None,
                 infeats = None,
                 nb_features = None,
                 nb_levels = None,
                 max_pool = 2,
                 feat_mult = 1,
                 nb_conv_per_level = 1,
                 half_res = False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor = s, mode = 'nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim = 1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LightningModule):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    ModalDict = {'src': 'acid', 'trg': 'iodine'}

    def __init__(self,
                 inshape: Sequence[int],
                 nb_unet_features: Optional[Union[int, Sequence[Sequence[int]]]] = None,
                 nb_unet_levels: Optional[int] = None,
                 unet_feat_mult: Optional[int] = 1,
                 nb_unet_conv_per_level: Optional[int] = 1,
                 int_steps: Optional[int] = 7,
                 int_downsize: Optional[int] = 2,
                 bidir: Optional[bool] = False,
                 use_probs: Optional[bool] = False,
                 src_feats: Optional[int] = 1,
                 trg_feats: Optional[int] = 1,
                 unet_half_res: Optional[bool] = False,
                 loss_config: Optional[Mapping[str, Any]] = None,
                 optimizer_config: Optional[Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]] = None):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats = (src_feats + trg_feats),
            nb_features = nb_unet_features,
            nb_levels = nb_unet_levels,
            feat_mult = unet_feat_mult,
            nb_conv_per_level = nb_unet_conv_per_level,
            half_res = unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size = 3, padding = 1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

        self.loss_config = loss_config
        self.loss_img = losses.__dict__[self.loss_config['img']['type']]()
        self.loss_grad = losses.Grad(self.loss_config['grad']['type'], int_downsize)
        self.loss_seg = losses.Dice()

        self.optimizer_config = optimizer_config if isinstance(optimizer_config, list) else [optimizer_config]
        if len(self.optimizer_config) == 1 and 'lr' in self.optimizer_config[0]['optimizer']:
            self.lr = self.optimizer_config[0]['optimizer']['lr']

    def __construct_optimizer(self, optimizer, set_lr = False):
        """
        Constructs the optimizer.

        Args:
            optimizer: dictionary containing optimizer configuration.
        """
        optimizer_type = optimizer.pop('type')
        if hasattr(self, 'lr') and self.lr is not None and set_lr:
            optimizer['lr'] = self.lr
        optimizer = optim.__dict__[optimizer_type](self.parameters(), **optimizer)

        return optimizer

    @staticmethod
    def __construct_lr_scheduler(optimizer, lr_scheduler):
        """
        Constructs the lr_scheduler.

        Args:
            optimizer: the optimizer used to construct the lr_scheduler.
            lr_scheduler: dictionary containing lr_scheduler configuration.
        """
        lr_scheduler_type = lr_scheduler.pop('type')
        lr_scheduler = optim.lr_scheduler.__dict__[lr_scheduler_type](optimizer, **lr_scheduler)

        return lr_scheduler

    def configure_optimizers(self):
        """
        Configure optimizers for model.
        """
        optimizer_configs = self.optimizer_config.copy()
        for idx, optimizer_config in enumerate(optimizer_configs):
            if 'optimizer' in optimizer_config:
                optimizer_config['optimizer'] = self.__construct_optimizer(optimizer_config['optimizer'], set_lr = idx == 0)
            if 'lr_scheduler' in optimizer_config and 'scheduler' in optimizer_config['lr_scheduler']:
                optimizer_config['lr_scheduler']['scheduler'] = self.__construct_lr_scheduler(optimizer_configs[idx]['optimizer'],
                                                                                              optimizer_config['lr_scheduler']['scheduler'])

        return optimizer_configs

    def forward(self, batch):
        # concatenate inputs and propagate unet
        source = batch[self.ModalDict['src']]['img']
        target = batch[self.ModalDict['trg']]['img']
        x = torch.cat([source, target], dim = 1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        if 'gt_segments_from_bboxes' in batch[self.ModalDict['src']]:
            segments = batch[self.ModalDict['src']]['gt_segments_from_bboxes']
            if isinstance(segments, list):
                y_seg = [self.transformer(segments[i][None, ...], pos_flow[i, None, ...]) for i in range(len(segments))]
            else:
                y_seg = self.transformer(segments, pos_flow)
        else:
            y_seg = None
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        res = {'y_source': y_source, 'pos_flow': pos_flow, 'preint_flow': preint_flow}
        if self.bidir:
            res['y_target'] = y_target
        if y_seg is not None:
            res['y_seg'] = y_seg

        return res

    def loss_step(self, batch, res, prefix = 'train'):
        loss = {}
        loss['img'] = self.loss_img(res['y_source'], batch[self.ModalDict['trg']]['img'])
        loss['grad'] = self.loss_grad(res['preint_flow'])
        if 'y_seg' in res:
            if isinstance(res['y_seg'], list):
                loss['seg'] = torch.mean(torch.stack(
                    [self.loss_seg(res['y_seg'][i], batch[self.ModalDict['trg']]['gt_segments_from_bboxes'][i][None, ...]) for i in
                     range(len(res['y_seg']))]))
            else:
                loss['seg'] = self.loss_seg(res['y_seg'], batch[self.ModalDict['trg']]['gt_segments_from_bboxes'])

        # multi loss weights
        loss = {k: v * (self.loss_config[k]['weight'] if 'weight' in self.loss_config[k] else 1) for k, v in loss.items()}

        # calculate loss
        loss['loss'] = torch.sum(torch.stack([v for v in loss.values()]))

        # add prefix
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: v for k, v in loss.items()}

        return loss

    def training_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        self.log_dict(loss)
        return loss['train/loss']

    def validation_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val')
        self.log_dict(loss)
        return loss

    def test_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test')
        self.log_dict(loss)
        return loss
