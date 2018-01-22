import torch
from torch.nn import NLLLoss2d, MSELoss


class BCELoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = torch.nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):

        probs_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        return self.bce_loss(probs_flat, targets_flat)


# FIXME: not working with weights
class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, batchsize=1):
        super(CrossEntropyLoss2d, self).__init__()
        self.batch_size = batchsize
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight, size_average, ignore_index)

    def forward(self, logits, targets):
        probs_flat = logits.view(self.batch_size, -1)
        targets_flat = targets.view(self.batch_size, -1).long()

        return self.cross_entropy_loss(logits, targets.long().squeeze(1))


# FIXME: not stable yet
class DiceLoss_old(torch.nn.Module):
    def __init__(self, smooth_const=1e-4):
        super(DiceLoss_old, self).__init__()
        self.smooth_const = smooth_const

    def forward(self, input_var, target_var):
        _input_var = input_var.clone()
        _target_var = target_var.clone()
        # _input_var.data = torch.mul(torch.add(_input_var.data, 1), 0.5).float()
        # _target_var.data = torch.mul(torch.add(_target_var.data, 1), 0.5).float()

        result_var = _input_var.clone()
        result_data = torch.mul(_input_var.data, _target_var.data).float()
        result_var.data = result_data

        numerator = result_var.sum().mul(2).add(self.smooth_const)
        denominator = _input_var.sum().add(_target_var.sum().add(self.smooth_const))
        dice_coeff = numerator.div(denominator)

        return 1-dice_coeff


class GANLoss(object):
    """Class to calculate GAN Loss"""
    def __init__(self, loss_fkt=torch.nn.MSELoss, tensor=torch.FloatTensor):
        """
        Function to create and initialize class variables
        :param loss_fkt: function to calculate losses
        :param tensor: Tensor type
        """
        super(GANLoss, self).__init__()
        self.real_label_value = 1.0
        self.fake_label_value = 0.0
        self.Tensor = tensor
        self.loss = loss_fkt()
        self.real_label = None
        self.fake_label = None

    def get_target_tensor(self, input_tensor, target_is_real):
        """
        Function to get the target-tensor (a Tensor of 1s if target is real, a Tensor of 0s otherwise)
        :param input_tensor: the input tensor the target-tensor should be compared with
        :param target_is_real: True if target is real, False otherwise
        :return: target tensor
        """

        target_tensor = None

        if target_is_real:
            create_label = ((self.real_label is None) or
                            (self.real_label.numel() != input_tensor.numel()))

            # No labels created yet or input dim does not match label dim
            if create_label:
                real_tensor = self.Tensor(input_tensor.size()).fill_(self.real_label_value)
                self.real_label = torch.autograd.Variable(real_tensor, requires_grad=False)

            target_tensor = self.real_label

        else:
            # No labels created yet or input dim does not match label dim
            create_label = ((self.fake_label is None) or
                            (self.fake_label.numel() != input_tensor.numel()))

            if create_label:
                fake_tensor = self.Tensor(input_tensor.size()).fill_(self.fake_label_value)
                self.fake_label = torch.autograd.Variable(fake_tensor, requires_grad=False)

            target_tensor = self.fake_label

        return target_tensor

    def __call__(self, input_tensor, target_is_real):
        """
        Function to make class callable
        :param input_tensor: input tensor (result of prediction)
        :param target_is_real: (True if input_tensor is real, False otherwise
        :return: loss value
        """

        target_tensor = self.get_target_tensor(input_tensor, target_is_real)

        # adding sigm when BCELoss is used is not necessary in Original CycleGAN, check why
        # if isinstance(self.loss, torch.nn.BCELoss):
        # sigm = torch.nn.Sigmoid()
        # input_tensor = sigm(input_tensor)
        return self.loss(input_tensor, target_tensor)
