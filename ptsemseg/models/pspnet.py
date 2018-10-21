import torch
import numpy as np
import torch.nn as nn

from ptsemseg.models.utils import *
from ptsemseg.loss import *

pspnet_specs = {
    'tgs': 
    {
         'n_classes': 2,
         'input_size': (101, 101),
         'block_config': [3, 3, 3],
    },
}

class pspnet(nn.Module):
    def __init__(self, 
                 n_classes=2, 
                 block_config=[3, 3, 3], 
                 input_size=(101,101), 
                 version=None, f_scale=1, p=0.1):

        super(pspnet, self).__init__()

        self.block_config = pspnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = pspnet_specs[version]['input_size'] if version is not None else input_size

        self.dropout = nn.Dropout2d(p=p)

        self.convbnrelu0 = conv2DBatchNormRelu(in_channels=1, n_filters=16*f_scale, k_size=3, padding=1, stride=1, bias=False) # Encoder first conv layers

        # (Dilated) Residual Blocks
        self.res_block1 = residualBlockPSP(n_blocks=self.block_config[0], in_channels=16*f_scale, mid_channels=8*f_scale, out_channels=32*f_scale, stride=2, dilation=1)
        self.res_block2 = residualBlockPSP(n_blocks=self.block_config[1], in_channels=32*f_scale, mid_channels=16*f_scale, out_channels=64*f_scale, stride=2, dilation=1)
        self.res_block3 = residualBlockPSP(n_blocks=self.block_config[2], in_channels=64*f_scale, mid_channels=32*f_scale, out_channels=128*f_scale, stride=2, dilation=1)
        self.res_block4 = residualBlockPSP(n_blocks=self.block_config[2], in_channels=128*f_scale, mid_channels=32*f_scale, out_channels=128*f_scale, stride=1, dilation=2)
        self.res_block5 = residualBlockPSP(n_blocks=self.block_config[2], in_channels=128*f_scale, mid_channels=32*f_scale, out_channels=128*f_scale, stride=1, dilation=4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(128*f_scale, [6, 3, 2, 1])

        self.cbr_final = conv2DBatchNormRelu(in_channels=256*f_scale, n_filters=64*f_scale, k_size=3, padding=1, stride=1, bias=False) # Encoder final conv layers
        self.cbr_gum3 = conv2DBatchNormRelu(in_channels=64*f_scale, n_filters=8*f_scale, k_size=1, padding=0, stride=1, bias=False) # GUM projection conv layers

        # Prediction layers
        self.classification = nn.Conv2d(8*f_scale, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.cbr_aux = conv2DBatchNormRelu(in_channels=128*f_scale, n_filters=8*f_scale, k_size=3, padding=1, stride=1, bias=False)
        self.aux_cls = nn.Conv2d(8*f_scale, self.n_classes, 1, 1, 0)

        # Decoder
        self.exit2 = residualBlockPSP(n_blocks=self.block_config[2], in_channels=128*f_scale, mid_channels=16*f_scale, out_channels=32*f_scale, stride=1, dilation=1)
        self.exit1 = residualBlockPSP(n_blocks=self.block_config[1], in_channels=64*f_scale, mid_channels=8*f_scale, out_channels=16*f_scale, stride=1, dilation=1)
        self.exit0 = residualBlockPSP(n_blocks=self.block_config[0], in_channels=32*f_scale, mid_channels=4*f_scale, out_channels=8*f_scale, stride=1, dilation=1)

        # SCSE: Spatial and Channel 'Squeeze and Excitation'
        self.scse_en1 = scSE(32*f_scale, reduction=2, is_batchnorm=True)
        self.scse_en2 = scSE(64*f_scale, reduction=2, is_batchnorm=True)
        self.scse_en3 = scSE(128*f_scale, reduction=2, is_batchnorm=True)
        self.scse_pp = scSE(64*f_scale, reduction=2, is_batchnorm=True)
        self.scse_de2 = scSE(32*f_scale, reduction=2, is_batchnorm=True)
        self.scse_de1 = scSE(16*f_scale, reduction=2, is_batchnorm=True)
        self.scse_de0 = scSE(8*f_scale, reduction=2, is_batchnorm=True)

        # GUM: Guided Upsampling Module
        self.gum_x3 = gum(in0_channels=16*f_scale, in1_channels=64*f_scale)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        x0 = self.convbnrelu0(x)

        x1 = self.res_block1(x0) # /2
        x1 = self.scse_en1(x1)
        x2 = self.res_block2(x1) # /2
        x2 = self.scse_en2(x2)
        x3 = self.res_block3(x2) # /2
        x3 = self.scse_en3(x3)

        if self.training: # Auxiliary layers for training
            x3_aux = self.cbr_aux(x3)

            x3_aux = self.dropout(x3_aux)
            x_aux = self.aux_cls(x3_aux)
            x_aux = F.upsample(x_aux, size=x0.shape[2:], mode='bilinear', align_corners=True)

        x3 = self.res_block4(x3)
        x3 = self.res_block5(x3)

        x3 = self.pyramid_pooling(x3)
        x3 = self.cbr_final(x3)

        grid_x3, offset_x3 = self.gum_x3(x0, x3)
        out_x3 = F.grid_sample(x3, grid_x3 + offset_x3, mode='bilinear', padding_mode='zeros')
        out_x3 = self.cbr_gum3(out_x3)

        x3 = self.scse_pp(x3)
        x3 = F.upsample(x3, size=x2.shape[2:], mode='bilinear', align_corners=True) # *2

        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.exit2(x2)
        x2 = self.scse_de2(x2)
        x2 = F.upsample(x2, size=x1.shape[2:], mode='bilinear', align_corners=True) # *2

        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.exit1(x1)
        x1 = self.scse_de1(x1)
        x1 = F.upsample(x1, size=x0.shape[2:], mode='bilinear', align_corners=True) # *2

        x0 = torch.cat([x0, x1], dim=1)
        x0 = self.exit0(x0)
        x0 = x0 + out_x3
        x0 = self.scse_de0(x0)

        x0 = self.dropout(x0)
        out = self.classification(x0)

        if self.training:
            return (out, x_aux), offset_x3
        else: # eval mode
            return out, offset_x3
