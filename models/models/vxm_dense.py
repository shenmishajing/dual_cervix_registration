import os
import shutil

import cv2
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .base import LightningModule
from .. import losses
from . import layers

from typing import Sequence, Optional, Mapping, Any
from utils import get_log_dir


class VxmDense(LightningModule):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    ModalDict = {'src': 'acid', 'trg': 'iodine'}

    def __init__(self,
                 backbone: nn.Module,
                 inshape: Sequence[int],
                 final_nf: Optional[int] = 16,
                 int_steps: Optional[int] = 7,
                 int_downsize: Optional[int] = 2,
                 bidir: Optional[bool] = False,
                 use_probs: Optional[bool] = False,
                 unet_half_res: Optional[bool] = False,
                 loss_config: Optional[Mapping[str, Any]] = None,
                 optimizer_config: Optional[Mapping[str, Any]] = None):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        self.int_downsize = int_downsize
        super().__init__(loss_config = loss_config, optimizer_config = optimizer_config)

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.backbone = backbone

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(final_nf, ndims, kernel_size = 3, padding = 1)

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

    def _build_loss(self, cls_type, **kwargs):
        return losses.__dict__[cls_type](**kwargs)

    def _build_loss_grad(self, cls_type, **kwargs):
        return losses.__dict__[cls_type](loss_mult = self.int_downsize, **kwargs)

    def forward(self, batch):
        # concatenate inputs and propagate unet
        source = batch[self.ModalDict['src']]['img']
        target = batch[self.ModalDict['trg']]['img']
        x = torch.cat([source, target], dim = 1)
        x = self.backbone(x)

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

    def _loss_step(self, batch, res, prefix = 'train'):
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

        return loss

    def validation_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val', use_loss_weight = False)
        self.log_dict(loss)
        return loss

    def test_step(self, batch, batch_idx):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test', use_loss_weight = False)
        self.log_dict(loss)
        return loss

    def on_predict_start(self) -> None:
        self.norm_cfg = {
            'acid': {'mean': [122.95729064941406], 'std': [15.282942771911621]},
            'iodine': {'mean': [83.96733856201172], 'std': [20.51559829711914]},
        }
        for part in self.norm_cfg:
            for k in self.norm_cfg[part]:
                self.norm_cfg[part][k] = torch.tensor(self.norm_cfg[part][k]).to(self.device)[None, :, None, None]

        log_dir = os.path.dirname(os.path.dirname(self.trainer.predicted_ckpt_path))
        self.output_path = os.path.join(log_dir, 'visualization')
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

    def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
        res = self(batch)
        res_img = {}
        for part in self.ModalDict.values():
            res_img[part] = batch[part]['img'] * self.norm_cfg[part]['std'] + self.norm_cfg[part]['mean']
        res_img['res'] = res['y_source'] * self.norm_cfg[self.ModalDict['trg']]['std'] + self.norm_cfg[self.ModalDict['trg']]['mean']
        res_img = torch.cat([res_img[self.ModalDict['src']], res_img['res'], res_img[self.ModalDict['trg']]], dim = -1)
        res_img = res_img.add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for i in range(res_img.shape[0]):
            cur_name = batch['acid']['img_metas'][i]['ori_filename'].removesuffix('_2.jpg') + '.png'
            cv2.imwrite(os.path.join(self.output_path, cur_name), res_img[i])
        return res_img
