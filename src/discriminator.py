import os
import numpy as np
import torch

from blocks import DiscriminatorBlock


class DiscriminatorDummy(torch.nn.Module):

    def __init__(self, input_nc=1, gpu_ids=[]):
        super(DiscriminatorDummy, self).__init__()
        self.input_nc = input_nc
        self.gpu_ids = gpu_ids

    def _build(self):
        pass

    def forward(self, input_tensor):
        return input_tensor


class ImageDiscriminator(torch.nn.Module):
    def __init__(self, input_nc=1, initial_filters=4, dropout_value=0.25, kernel_size=3, strides=2, gpu_ids=[]):
        super(ImageDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.initial_filters = initial_filters
        self.dropout_value = dropout_value
        self.kernel_size = kernel_size
        self.strides = strides

        self.gpu_ids = gpu_ids
        self._build()
        if len(gpu_ids):
            self.first_model_part.cuda(device_id=gpu_ids[0])
            self.second_model_part.cuda(device_id=gpu_ids[0])

    def _build(self):

        model = [DiscriminatorBlock(self.input_nc, self.dropout_value, self.initial_filters, self.kernel_size,
                                    self.strides)]
        i = 1
        while i <= 8:
            i *= 2
            model += [DiscriminatorBlock(int((i/2))*self.initial_filters, self.dropout_value, i*self.initial_filters,
                                         self.kernel_size, self.strides)]

        # model += [torch.nn.AvgPool2d(self.input_size)]

        model += [torch.nn.AdaptiveAvgPool2d(1)]
        self.first_model_part = torch.nn.Sequential(*model)

        model = [torch.nn.Linear(i*self.initial_filters, 1),
                 torch.nn.Sigmoid()]
        self.second_model_part = torch.nn.Sequential(*model)

    def forward(self, input_tensor):
        if self.gpu_ids and isinstance(input_tensor.data, torch.cuda.FloatTensor):
            tmp = torch.nn.parallel.data_parallel(self.first_model_part, input_tensor, self.gpu_ids)
            return torch.nn.parallel.data_parallel(self.second_model_part, tmp.view(tmp.size(0), -1), self.gpu_ids)
        else:
            tmp = self.first_model_part(input_tensor)
            return self.second_model_part(torch.squeeze(tmp))


class ImageDiscriminatorConv(torch.nn.Module):
    def __init__(self, input_nc=1, initial_filters=64, dropout_value=0.25, n_blocks=4, gpu_ids=[]):
        super(ImageDiscriminatorConv, self).__init__()
        self.input_nc = input_nc
        self.initial_filters = initial_filters
        self.dropout_value = dropout_value

        self.strides = 2
        self.n_blocks = n_blocks

        self.gpu_ids = gpu_ids
        self._build()
        if len(gpu_ids):
            self.model.cuda(device_id=gpu_ids[0])

    def _build(self):
        model = [DiscriminatorBlock(self.input_nc, self.dropout_value, self.initial_filters, 5, self.strides),
                 DiscriminatorBlock(self.initial_filters, self.dropout_value, 2*self.initial_filters, 5, self.strides)]

        filters = 2 * self.initial_filters

        i = 2
        for idx in range(2, self.n_blocks):
            filters_old = filters
            filters = max(2*filters, 512)
            model += [DiscriminatorBlock(filters_old, self.dropout_value, filters, 3, self.strides)]

        model += [torch.nn.Conv2d(filters, 1, 3, self.strides, 1)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input_tensor):
        if self.gpu_ids and isinstance(input_tensor.data, torch.cuda.FloatTensor):
            return torch.nn.parallel.data_parallel(self.model, input_tensor, self.gpu_ids)
        else:
            return self.model(input_tensor)
