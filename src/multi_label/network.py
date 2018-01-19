from collections import OrderedDict

import torch
import os
import numpy as np

from u_net import uNet
from discriminator import ImageDiscriminatorConv, ImageDiscriminator
from image_pool import ImagePool

from losses import GANLoss, BinarySelectiveCrossEntropyLoss


class Network(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_output_channels=1, n_blocks=9, initial_filters=64, dropout_value=0.25,
                 lr=1e-3, decay=0, decay_epochs=0, batch_size=1, image_width=640, image_height=640,
                 load_network=False, load_epoch=0, model_path='', name='', gpu_ids=[], gan=False,
                 pool_size=50, lambda_gan=1, n_blocks_discr=3):
        super(Network, self).__init__()
        self.input_nc = n_input_channels
        self.output_nc = n_output_channels
        self.n_blocks = n_blocks
        self.initial_filters = initial_filters
        self.dropout_value = dropout_value
        self.lr = lr
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.generator = torch.nn.Module()
        self.discriminator = torch.nn.Module()
        self.decay = decay
        self.decay_epochs = decay_epochs
        self.save_dir = model_path
        os.makedirs(self.save_dir, exist_ok=True)

        self.input_img = None
        self.input_gt = None
        self.var_img = None
        self.var_gt = None
        self.fake_mask = None
        self.dont_care_mask = None

        self.criterion_seg = None
        self.criterion_gan = None
        self.optimizer_seg = None
        self.optimizer_dis = None
        self.fake_mask_pool = None

        self.loss = None
        self.loss_seg = None
        self.loss_g = None
        self.loss_g_gan = None
        self.loss_d_gan = None
        self.gan = gan
        self.pool_size = pool_size
        self.lambda_gan = lambda_gan
        self.n_blocks_discr = n_blocks_discr

        self.load_network = load_network
        self.name = name
        self.load_epoch = load_epoch

        if len(gpu_ids):
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.FloatTensor

        self.initialize(n_input_channels, n_output_channels, n_blocks, initial_filters, dropout_value,
                        lr, batch_size, image_width, image_height, gpu_ids, gan, pool_size, n_blocks_discr)

    def cuda(self):
        self.generator.cuda()

    def initialize(self, n_input_channels, n_output_channels, n_blocks, initial_filters, dropout_value,
                   lr,  batch_size, image_width, image_height,  gpu_ids, gan, pool_size, n_blocks_discr):

        self.input_img = self.tensor(batch_size, n_input_channels, image_height, image_width)
        self.input_gt = self.tensor(batch_size, n_output_channels, image_height, image_width)

        self.generator = uNet(n_input_channels, n_output_channels, n_blocks, initial_filters, dropout_value, gpu_ids)

        if gan:
            self.discriminator = ImageDiscriminatorConv(n_output_channels, initial_filters, dropout_value,
                                                        gpu_ids=gpu_ids, n_blocks=n_blocks_discr)
            self.criterion_gan = GANLoss(tensor=self.tensor)
            self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            self.fake_mask_pool = ImagePool(pool_size)

        if self.load_network:
            self._load_network(self.generator, 'Model', self.load_epoch)
            if gan:
                self._load_network(self.discriminator, 'Discriminator', self.load_epoch)

        self.criterion_seg = BinarySelectiveCrossEntropyLoss()
        self.optimizer_seg = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        print('---------- Network initialized -------------')
        self.print_network(self.generator)
        if gan:
            self.print_network(self.discriminator)
        print('-----------------------------------------------')

    def set_input(self, input_img, input_gt=None):

        if input_img is not None:
            self.input_img.resize_(input_img.size()).copy_(input_img)

        if input_gt is not None:
            self.input_gt.resize_(input_gt.size()).copy_(input_gt)

    def forward(self, vol=False):
        """
        Function to create autograd variables of inputs (necessary for back-propagation)
        :param vol: True if no backprop is needed
        :return:
        """
        self.var_img = torch.autograd.Variable(self.input_img, volatile=vol)
        self.var_gt = torch.autograd.Variable(self.input_gt, volatile=vol)

    def predict(self):
        """
        Function to predict from datasets
        :return: fakeB: generated image from dataset A to look like images in dataset B
        :return: recA: reconstructed image from fakeB
        :return: fakeA: generated image from dataset B to look like images in dataset A
        :return: recB: reconstructed image from fakeA
        """
        assert (self.input_img is not None)

        self.var_img = torch.autograd.Variable(self.input_img, volatile=True)
        self.fake_mask = self.generator.forward(self.var_img)

        return self.fake_mask

    def backward_seg(self):
        self.fake_mask = self.generator.forward(self.var_img)

        self.loss_seg = self.criterion_seg(self.fake_mask, self.var_gt)

        self.loss_g = self.loss_seg

        if self.gan:
            pred_fake = self.discriminator.forward(self.fake_mask)
            self.loss_g_gan = self.criterion_gan(pred_fake, True)
            self.loss_g = self.loss_seg + self.loss_g_gan*self.lambda_gan

        self.loss_g.backward()

    def backward_d(self):
        fake_mask = self.fake_mask_pool.query(self.fake_mask)
        pred_real = self.discriminator.forward(self.var_gt)
        loss_d_real = self.criterion_gan(input_tensor=pred_real, target_is_real=True)
        pred_fake = self.discriminator.forward(fake_mask.detach())
        loss_d_fake = self.criterion_gan(input_tensor=pred_fake, target_is_real=False)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.loss_d_gan = loss_d

    def optimize(self):
        """
        Function for parameter optimization
        :return: None
        """

        self.forward()

        self.optimizer_seg.zero_grad()
        self.backward_seg()
        self.optimizer_seg.step()

        if self.gan:
            self.optimizer_dis.zero_grad()
            self.backward_d()
            self.optimizer_dis.step()

    def get_current_errors(self):
        """
        Function to get access to current errors outside class
        :return: OrderedDict with values different models
        """

        errors = [self.loss_seg.data[0]]
        labels = ["Seg"]

        if self.gan:
            errors.append(self.loss_d_gan.data[0])
            errors.append(self.loss_g_gan.data[0])
            errors.append(self.loss_g.data[0])
            labels.append("Discr")
            labels.append("Seg_GAN")
            labels.append("Seg_total")
        tuple_list = list(zip(labels, errors))

        return OrderedDict(tuple_list)

    def save(self, label):
        """
        Function to save the subnets
        :param label: label (part of the file the subnet will be saved to)
        :return: None
        """
        self._save_network(self.generator, 'Model', label, self.gpu_ids)
        if self.gan:
            self._save_network(self.discriminator, 'Discriminator', label, self.gpu_ids)

    def _save_network(self, network, network_label, epoch_label, gpu_ids):
        """
                Helper Function for saving pytorch networks (can be used in subclasses)
                :param network: the network to save
                :param network_label: the network label (name)
                :param epoch_label: the epoch to save
                :param gpu_ids: the gpu ids to continue training after saving
                :return: None
                """

        save_filename = str(self.name) + '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    def _load_network(self, network, network_label, epoch_label):
        """
        Helper Function for loading pytorch networks (can be used in subclasses)
        :param network: the network variable to store the loaded network in
        :param network_label: part of the filename the network should be loaded from
        :param epoch_label: the epoch to load
        :return: None
        """
        save_filename = str(self.name) + '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        """
        Function for learning rate scheduling
        :return: None
        """
        tmp = self.lr

        self.lr -= (self.decay/self.decay_epochs)
        # for param_group in self.optimizer_d.param_groups:
        #     param_group['lr'] = self.lr
        for param_group in self.optimizer_seg.param_groups:
            param_group['lr'] = self.lr

        if self.gan:
            for param_group in self.optimizer_dis.param_groups:
                param_group['lr'] = self.lr

        print('update learning rate: %f -> %f' % (tmp, self.lr))

    @staticmethod
    def print_network(network):
        """
        Static Helper Function to print a network summary
        :param network:
        :return: None
        """
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)


class NetworkBench(torch.nn.Module):

    def __init__(self, n_networks=5, n_input_channels=3, n_output_channels=1, n_blocks=9, initial_filters=64, dropout_value=0.25,
                 lr=1e-3, decay=0, decay_epochs=0, batch_size=1, image_width=640, image_height=640,
                 load_network=False, load_epoch=0, model_path='', name='', gpu_ids=[], gan=False,
                 pool_size=50, lambda_gan=1, n_blocks_discr=4):
        super(NetworkBench, self).__init__()
        self.models = []
        self.gan = gan
        self.n_networks = n_networks
        for i in range(n_networks):
            self.models.append(Network(n_input_channels, n_output_channels, n_blocks, initial_filters, dropout_value,
                                       lr, decay, decay_epochs, batch_size, image_width, image_height,
                                       load_network, load_epoch, model_path, name + "_%d" % i, gpu_ids, gan,
                                       pool_size, lambda_gan, n_blocks_discr))

    def cuda(self):
        for model in self.models:
            model.cuda()

    def set_inputs(self, input_imgs, input_gts=None):
        assert len(input_imgs) == self.n_networks

        if input_gts is not None:
            assert len(input_gts) == self.n_networks

            for idx, data in enumerate(list(zip(input_imgs, input_gts))):
                self.models[idx].set_input(data[0], data[1])

        else:
            for idx, img in enumerate(input_imgs):
                self.models[idx].set_input(img, None)

    def forward(self, vol=False):
        for model in self.models:
            model.forward(vol=vol)

    def predict(self):
        return [model.predict() for model in self.models]

    def backward(self):
        for model in self.models:
            model.backward()

    def backward_d(self):
        if self.gan:
            for model in self.models:
                model.backward_d()

    def optimize(self):
        for model in self.models:
            model.optimize()

    def get_current_errors(self):
        labels, errors = [], []

        for idx, model in enumerate(self.models):
            error_dict = model.get_current_errors()
            for key, value in error_dict.items():
                labels.append(str(key) + "_%d" % idx)
                errors.append(value)

        return OrderedDict(list(zip(labels, errors)))

    def save(self, label):
        for model in self.models:
            model.save(label)

    def update_learning_rate(self):
        for model in self.models:
            model.update_learning_rate()

    def get_current_imgs(self, model_idx=None):
        out_list = []
        if model_idx is None or model_idx > len(self.models):
            for model in self.models:
                model_dict = {'img': model.var_img,
                              'mask': model.fake_mask,
                              'gt': model.var_gt}
                out_list.append(model_dict)

        else:
            with self.models[model_idx] as model:
                model_dict = {'img': model.var_img,
                              'mask': model.fake_mask,
                              'gt': model.var_gt}
                out_list.append(model_dict)

        return out_list
