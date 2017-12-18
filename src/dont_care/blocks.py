import torch
from torchvision.models.inception import BasicConv2d


class BaseBlock(torch.nn.Module):
    """class to define an API for all Layer Blocks"""
    def __init__(self, outer_nc, dropout_value):
        """
        function to create and initialize class variables
        :param outer_nc: number of channels outside the block (used as block-input and -output)
        :param dropout_value: dropout-value
        """
        super(BaseBlock, self).__init__()
        self.outer_nc = outer_nc
        self.dropout_value = dropout_value
        self.model = torch.nn.Sequential()

    def _build(self):
        """
        Abstract Function to build the block, must be implemented in subclasses
        :return: None
        """
        pass

    def forward(self, input_tensor):
        """
        Abstract Function to forward through a block (necessary for implicit backward function),
        must be implemented in subclasses

        :param input_tensor: input tensor of the block
        :return: forwarded result
        """
        return self.model(input_tensor)


class UnetSkipConnectionBlock(BaseBlock):
    """class containing a U Skip Connection Block implementation"""
    def __init__(self, outer_nc=64, dropout_value=0.25, inner_nc=64, submodule=None,
                 outermost=False, innermost=False):
        """
        function to create and initialize the class variables
        :param outer_nc: number of input and output channels for block
        :param dropout_value: dropout value
        :param inner_nc: number of channels given to submodule
        :param submodule: submodule inside the unet block
        :param outermost: True if block is directly connected to input image, False otherwise
        :param innermost: True if block contains no submodule
        """
        super(UnetSkipConnectionBlock, self).__init__(outer_nc, dropout_value)
        self.inner_nc = inner_nc
        self.submodule = submodule
        self.outermost = outermost
        self.innermost = innermost
        self._build()

    def _build(self):
        """
        function to build the block
        :return: None
        """
        downconv = torch.nn.Conv2d(self.outer_nc, self.inner_nc, kernel_size=4,
                                   stride=2, padding=1)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.BatchNorm2d(self.inner_nc, affine=True)
        uprelu = torch.nn.ReLU(True)
        upnorm = torch.nn.BatchNorm2d(self.outer_nc, affine=True)

        if self.outermost:
            upconv = torch.nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
                                              kernel_size=4, stride=2,
                                              padding=1)
            down = [torch.nn.Dropout(self.dropout_value), downconv]

            # up = [uprelu, torch.nn.Dropout(self.dropout_value), upconv, torch.nn.Tanh()]
            up = [uprelu, torch.nn.Dropout(self.dropout_value), upconv]

            model = down + [self.submodule] + up
        elif self.innermost:
            upconv = torch.nn.ConvTranspose2d(self.inner_nc, self.outer_nc,
                                              kernel_size=4, stride=2,
                                              padding=1)
            down = [downrelu, torch.nn.Dropout(self.dropout_value), downconv]
            up = [uprelu, torch.nn.Dropout(self.dropout_value), upconv, upnorm]
            model = down + up
        else:
            upconv = torch.nn.ConvTranspose2d(self.inner_nc * 2, self.outer_nc,
                                              kernel_size=4, stride=2,
                                              padding=1)
            down = [downrelu, torch.nn.Dropout(self.dropout_value), downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [self.submodule] + up + [torch.nn.Dropout(self.dropout_value)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input_tensor):
        """
        Function to forward through a block (necessary for implicit backward function)
        :param input_tensor: input tensor
        :return: None
        """
        if self.outermost:
            return self.model(input_tensor)
        else:
            out = torch.cat([self.model(input_tensor), input_tensor], 1)
            return out


class BasicConv2dTranspose(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dTranspose, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool, indices = self.maxpool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1), indices


class InceptionTransposeBlockB(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionTransposeBlockB, self).__init__()

        self.branch3x3 = BasicConv2dTranspose(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2dTranspose(96, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2dTranspose(96, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2dTranspose(in_channels, 96, kernel_size=3, stride=2)

        self.unpool = torch.nn.MaxUnpool2d(kernel_size=3, stride=2)

    def forward(self, x, indices):
        branch_unpool = self.unpool(x, indices)

        branch_dbl = self.branch3x3dbl_3(x)
        branch_dbl = self.branch3x3dbl_2(branch_dbl)
        branch_dbl = self.branch3x3dbl_1(branch_dbl)

        branch3x3 = self.branch3x3(x)

        outputs = [branch3x3, branch_dbl, branch_unpool]
        return torch.cat(outputs, 1)


class UnetSkipConnectionInceptionBlock(BaseBlock):
    def __init__(self, outer_nc=64, dropout_value=0.25, inner_nc=64, submodule=None,
                 outermost=False, innermost=False):
        super(UnetSkipConnectionInceptionBlock, self).__init__(outer_nc, dropout_value)
        self.inner_nc = inner_nc
        self.submodule = submodule
        self.outermost = outermost
        self.innermost = innermost
        self._build()

    def _build(self):
        self.down_block = InceptionBlock(self.inner_nc)
        self.down_conv0 = torch.nn.Conv2d(self.outer_nc, self.inner_nc, kernel_size=1)
        down_conv1 = [torch.nn.Dropout(self.dropout_value), torch.nn.Conv2d(self.inner_nc + 480, self.inner_nc, kernel_size=1)]
        self.down_conv1 = torch.nn.Sequential(*down_conv1)
        self.up_block = InceptionBlockTranspose(self.inner_nc*2)
        up_conv = [torch.nn.Dropout(self.dropout_value), torch.nn.Conv2d(self.inner_nc*2 + 448, self.outer_nc, kernel_size=1)]
        self.up_conv = torch.nn.Sequential(*up_conv)

        if self.innermost:
            self.up_block = InceptionBlockTranspose(self.inner_nc)
            up_conv = [torch.nn.Dropout(self.dropout_value),
                       torch.nn.Conv2d(self.inner_nc + 448, self.outer_nc, kernel_size=1)]
            self.up_conv = torch.nn.Sequential(*up_conv)

    def forward(self, input_tensor):
        """
        Function to forward through a block (necessary for implicit backward function)
        :param input_tensor: input tensor
        :return: None
        """

        down = self.down_conv0(input_tensor)
        down, indices = self.down_block(down)
        down = self.down_conv1(down)

        if self.innermost:
            intermediate = down
            _indices = indices
        else:
            intermediate = torch.cat([self.submodule(down), down], dim=1)
            _indices = torch.cat([indices, indices], dim=1)

        up = self.up_block(intermediate, _indices)
        up = self.up_conv(up)

        return up


class DiscriminatorBlock(BaseBlock):
    def __init__(self, input_nc, dropout_value, n_filters, kernel_size, strides):
        super(DiscriminatorBlock, self).__init__(input_nc, dropout_value)
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.strides = strides
        # _padding_size = (input_img_size*strides - strides - input_img_size + kernel_size)/2
        _padding_size = (kernel_size - 1) / 2
        self.input_padding = int(_padding_size)
        # if _padding_size % 2 != 0:
        #     self.output_padding = (1, 0)
        # else:
        #     self.output_padding = 0

        self._build()

    def _build(self):
        model = [torch.nn.Conv2d(self.outer_nc, self.n_filters, self.kernel_size, self.strides, self.input_padding),
                 # torch.nn.ZeroPad2d(self.output_padding),
                 torch.nn.BatchNorm2d(self.n_filters),
                 torch.nn.ReLU(),
                 torch.nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding=self.input_padding),
                 # torch.nn.ZeroPad2d(self.output_padding),
                 torch.nn.BatchNorm2d(self.n_filters),
                 torch.nn.ReLU(),
                 torch.nn.Dropout(self.dropout_value),
                 torch.nn.MaxPool2d(self.strides)
                 ]
        self.model = torch.nn.Sequential(*model)

    def forward(self, input_tensor):
        return self.model(input_tensor)


InceptionBlockTranspose = InceptionTransposeBlockB

