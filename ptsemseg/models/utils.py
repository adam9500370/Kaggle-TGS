import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, is_batchnorm=True):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, negative_slope=0.0, is_batchnorm=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)
        relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if negative_slope > 0.0 else nn.ReLU(inplace=True)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          relu_mod,)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          relu_mod,)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, is_batchnorm=True):
        super(deconv2DBatchNorm, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                      padding=padding, stride=stride, bias=bias)

        if is_batchnorm:
            self.dcb_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),)
        else:
            self.dcb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, negative_slope=0.0, is_batchnorm=True):
        super(deconv2DBatchNormRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                      padding=padding, stride=stride, bias=bias)
        relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if negative_slope > 0.0 else nn.ReLU(inplace=True)

        if is_batchnorm:
            self.dcbr_unit = nn.Sequential(conv_mod,
                                           nn.BatchNorm2d(int(n_filters)),
                                           relu_mod,)
        else:
            self.dcbr_unit = nn.Sequential(conv_mod,
                                           relu_mod,)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class pyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, is_batchnorm=True):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, is_batchnorm=is_batchnorm))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        h, w = x.shape[2:]

        output_slices = [x]
        for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
            stride = (int(h/pool_size), int(w/pool_size))
            k_size = (int(h - stride[0]*(pool_size-1)), int(w - stride[1]*(pool_size-1)))
            out = F.avg_pool2d(x, k_size, stride=stride, padding=0)
            out = module(out)
            out = F.upsample(out, size=(h, w), mode='bilinear', align_corners=True)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1, groups=1, is_batchnorm=True):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=stride, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x) if self.in_channels != self.out_channels else x
        return F.relu(conv+residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, groups=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=1, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x+residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, groups=1, is_batchnorm=True):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        layers = []
        layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm))
        for i in range(n_blocks-1):
            layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnormn))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class cSE(nn.Module): # Spatial Squeeze and Channel Excitation
    def __init__(self, in_channels, reduction=2, is_batchnorm=False):
        super(cSE, self).__init__()

        bias = not is_batchnorm

        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
                                 conv2DBatchNormRelu(in_channels, in_channels // reduction, k_size=1, stride=1, padding=0, bias=bias, dilation=1, negative_slope=0.0, is_batchnorm=is_batchnorm),
                                 conv2DBatchNorm(in_channels // reduction, in_channels, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x)


class sSE(nn.Module): # Channel Squeeze and Spatial Excitation
    def __init__(self, in_channels, is_batchnorm=False):
        super(sSE, self).__init__()

        bias = not is_batchnorm

        self.sse = nn.Sequential(conv2DBatchNorm(in_channels, 1, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.sse(x)


class scSE(nn.Module): # Concurrent Spatial and Channel 'Squeeze and Excitation'
    def __init__(self, in_channels, reduction=2, is_batchnorm=False):
        super(scSE, self).__init__()

        self.cse_module = cSE(in_channels, reduction=reduction, is_batchnorm=is_batchnorm)
        self.sse_module = sSE(in_channels, is_batchnorm=is_batchnorm)

    def forward(self, x):
        return self.cse_module(x) + self.sse_module(x)


class gum(nn.Module): # Guided Upsampling Module
    def __init__(self, in0_channels, in1_channels, is_batchnorm=True):
        super(gum, self).__init__()

        bias = not is_batchnorm

        self.cb_0 = conv2DBatchNorm(in0_channels, in1_channels, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm) # projection
        self.cb_1 = conv2DBatchNorm(in1_channels, in1_channels, k_size=3, stride=1, padding=2, bias=bias, dilation=2, is_batchnorm=is_batchnorm) # dilated conv

        self.cbr_1 = conv2DBatchNormRelu(in1_channels, in1_channels // 2, k_size=3, stride=1, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm)
        self.cbr_2 = conv2DBatchNormRelu(in1_channels // 2, in1_channels // 4, k_size=3, stride=1, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm)
        self.cbr_3 = conv2DBatchNorm(in1_channels // 4, 2, k_size=3, stride=1, padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm)

    def forward(self, x0, x1):
        n, c, oh, ow = x0.size()

        offset_norm = torch.zeros(n, oh, ow, 2, device=torch.device('cuda')).float()
        offset_norm[:, :, :, 0] = (ow - 1) / 2
        offset_norm[:, :, :, 1] = (oh - 1) / 2
        offset_norm = offset_norm.detach()

        x1 = F.upsample(x1, size=x0.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.cb_0(x0)
        x1 = self.cb_1(x1)
        x = F.relu(x0 + x1, inplace=True)

        offset = self.cbr_3(self.cbr_2(self.cbr_1(x))) # (N, 2, H, W)
        offset = offset.transpose_(1, 2).transpose_(2, 3)
        offset = offset / offset_norm

        # normalize to [-1, 1]
        h = torch.arange(0, oh, device=torch.device('cuda')).float() / (oh-1) * 2 - 1
        w = torch.arange(0, ow, device=torch.device('cuda')).float() / (ow-1) * 2 - 1

        grid = torch.zeros(oh, ow, 2, device=torch.device('cuda')).float()
        grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
        grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).detach() # (regular) grid.shape: [n, oh, ow, 2]

        return grid, offset # (N, H, W, 2)
