from shutil import copyfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from network import NetworkBench
from image_handler import ImageHandler
import os
import numpy as np

import time

from correspondence_data_loader import CorrespondenceDataLoader, CorrespondenceDataLoaderDontCare, DataLoaderDiscriminator

from tensorboardX import SummaryWriter

class SegmentationNetwork(object):

    def __init__(self, options, gpu_ids=[]):
        self.options = options
        self.model = NetworkBench(n_networks=options.n_networks,
                                  n_input_channels=options.input_nc,
                                  n_output_channels=options.output_nc,
                                  n_blocks=options.n_blocks,
                                  initial_filters=options.initial_filters,
                                  dropout_value=options.dropout_value,
                                  lr=options.lr,
                                  decay=options.decay,
                                  decay_epochs=options.decay_epochs,
                                  batch_size=options.batch_size,
                                  image_width=options.image_width,
                                  image_height=options.image_height,
                                  load_network=options.load_network,
                                  load_epoch=options.load_epoch,
                                  model_path=os.path.join(options.model_path, options.name),
                                  name=options.name,
                                  gpu_ids=gpu_ids,
                                  dont_care=options.dont_care,
                                  gan=options.gan,
                                  pool_size=options.pool_size,
                                  lambda_gan=options.lambda_gan,
                                  n_blocks_discr=options.n_blocks_discr)

        self.model.cuda()

        self.dont_care = options.dont_care
        self.gan = options.gan

        if self.gan:
            self.discriminator_datasets = DataLoaderDiscriminator(options).load_data()

        self.data_sets = CorrespondenceDataLoaderDontCare(options).load_data()
        self.image_handler = ImageHandler()
        self.loss_dir = self.options.output_path + "/" + self.options.name + "/Train"
        copyfile(os.path.relpath('seg_config.yaml'), os.path.join(self.options.model_path, self.options.name, 'seg_config.yaml'))
        self.writer = SummaryWriter(self.loss_dir)

    def train(self):
        """
        Function to train model from dataset
        :return: None
        """
        print("Started Training")
        batch_size = self.options.batch_size
        loss_file = self.loss_dir + "/losses.txt"

        if os.path.isfile(loss_file):
            if self.options.load_network:
                self.erase_loss_file(loss_file, self.options.load_epoch)
            else:
                self.erase_loss_file(loss_file, 0)

        if self.options.load_network:
            base_epoch = int(self.options.load_epoch)
        else:
            base_epoch = 0

        for epoch in range(1 + base_epoch, self.options.n_epochs + self.options.decay_epochs + 1 + base_epoch):

            epoch_start_time = time.time()

            steps = 0
            t = 0

            # Get Iteraters for each dataset
            data_iters = []
            for loader in self.data_sets:
                data_iters.append(iter(loader))

            discr_data_iters = []
            if self.gan:
                for loader in self.discriminator_datasets:
                    discr_data_iters.append(iter(loader))

            for i in range(len(self.data_sets[0])):
                iter_start_time = time.time()

                current_batch_imgs, current_batch_labels, dont_care_masks, discriminator_imgs = [], [], [], []

                for iterator in data_iters:
                    data = next(iterator)
                    current_batch_imgs.append(data['img'])
                    current_batch_labels.append(data['label'])
                    if self.dont_care:
                        dont_care_masks.append(data['dont_care'])
                    else:
                        dont_care_masks = None
                if self.gan:
                    for idx, iterator in enumerate(discr_data_iters):
                        try:
                            discriminator_imgs.append(next(iterator))
                        except StopIteration:
                            discr_data_iters[idx] = iter(self.discriminator_datasets[idx])
                            discriminator_imgs.append(next(discr_data_iters[idx]))

                self.model.set_inputs(current_batch_imgs, current_batch_labels, dont_care_masks, discriminator_imgs)

                self.model.optimize()

                if (steps + 1) % self.options.print_freq == 0:
                    errors = self.model.get_current_errors()
                    t = (time.time() - iter_start_time)

                    # with open(loss_file, 'a+') as f:
                    #     f.write(message + "\n")

                    message = '(epoch: %d, step: %d, time/step: %.3f)\n' % (epoch, steps + 1, t)
                    for k, v in errors.items():
                        message += '%s: %.3f, ' % (k, float(v))

                    if not os.path.isdir(str(self.loss_dir)):
                        os.makedirs(str(self.loss_dir))

                    for k, v in errors.items():
                        self.writer.add_scalar('Model %d/%s' % (float(k.split("_")[-1]), k.split("_")[0]), float(v), (epoch-1)*self.options.steps_per_epoch + steps)

                    print(message)
                    # self.plot_losses(loss_file)

                steps += 1
                if steps >= self.options.steps_per_epoch:
                    break

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.options.n_epochs + self.options.decay_epochs, time.time() - epoch_start_time))

            if epoch % self.options.save_img_freq == 0:
                # self.model.predict()
                output_dir = self.options.output_path + "/" + self.options.name + "/Train/images"
                img_list = self.model.get_current_imgs()
                for idx, images in enumerate(img_list):
                    self.image_handler.save_image(images['img'], output_dir, 'epoch_%03d_real_img_model_%d' % (epoch, idx))
                    self.image_handler.save_mask(images['mask'], output_dir, 'epoch_%03d_fake_mask_model_%d' % (epoch, idx))
                    self.image_handler.save_mask(images['gt'], output_dir, 'epoch_%03d_gt_model_%d' % (epoch, idx))
                    self.writer.add_image("Real Images", images['img'].data, (epoch-1)*self.options.steps_per_epoch + steps)
                    self.writer.add_image("fake Masks", images['mask'].data, (epoch-1)*self.options.steps_per_epoch + steps)
                    self.writer.add_image("Groundtruth Masks", images['gt'].data, (epoch-1)*self.options.steps_per_epoch + steps)

                self.create_html_file(epoch)

            if epoch % self.options.save_freq == 0:
                print('saving the model at the end of epoch %d' % epoch)
                self.model.save(str(epoch))

            if epoch > self.options.n_epochs:
                self.model.update_learning_rate()

    def predict_to_dir(self, n_predicts=0):
        """
        Function to predict Images from Dataroot to a subfolder of output_path
        :param n_predicts: number of Images to predict, set 0 to predict all images
        :return:
        """

        print("Started Prediction")
        if not n_predicts:
            n_predicts = max([len(dataset) for dataset in self.data_sets])

        for i, data in enumerate(self.data_sets[0]):
            self.model.set_inputs([data['img'] for x in range(self.options.n_networks)])
            predicted_mask = self.model.predict()

            # FIXME: data['path_img'] gives list of strings instead of string
            save_name = (os.path.split(data['path_img'][0])[-1]).rsplit('.', 1)[0]

            output_dir = self.options.output_path + "/" + self.options.name + "/Predict/" + "Epoch_" + str(self.options.load_epoch)
            self.image_handler.save_mask(predicted_mask[0], output_dir, save_name + '_pred')
            self.image_handler.save_image(self.model.get_current_imgs()[0]['img'], output_dir, save_name)

            if ((i + 1) % 10) == 0:
                print("Predicted %d of %d Images" % (i+1, n_predicts))

            if (i + 1) >= n_predicts:
                break

        print("Finished Prediction")

    def predict(self, n_predicts=0):

        print("Started Prediction")

        predictions = []
        save_names = []

        if not n_predicts:
            n_predicts = max([len(dataset) for dataset in self.data_sets])

        for i, data in enumerate(self.data_sets[0]):
            self.model.set_inputs([data['img'] for x in range(self.options.n_networks)])
            predicted_mask = self.model.predict()
            save_name = (os.path.split(data['path_img'][0])[-1]).rsplit('.', 1)[0] + '_pred'
            save_names.append(save_name)

            predictions.append(predicted_mask)

            if ((i + 1) % 10) == 0:
                print("Predicted %d of %d Images" % (i+1, n_predicts))

            if (i + 1) >= n_predicts:
                break

        print("Finished Prediction")
        return predictions, save_names

    def get_annotation_suggestions(self, n_predicts=0, n_samples=100):

        n_digits = len(repr(abs(n_samples)))

        predictions, save_names = self.predict(n_predicts=n_predicts)

        images = [[ImageHandler._tensor_to_image(img, mask=True) for img in imgs] for imgs in predictions]
        # sum_images = np.sum(images, axis=0)
        result_imgs = []
        for imgs in images:
            result_and = np.zeros_like(imgs[0])
            result_or = np.zeros_like(imgs[0])
            for img in imgs:
                result_and = np.logical_and(img, result_and)
                result_or = np.logical_or(img, result_or)

            result_imgs.append(result_or - result_and)

        uncertainties = []
        for idx, res_img in result_imgs:
            uncertainties.append((np.sum(res_img), idx))

        uncertainties.sort(key=lambda tup: tup[0], reverse=True)

        if n_samples < len(uncertainties):
            n_samples = len(uncertainties)

        output_dir = self.options.output_path + "/" + self.options.name + "/Suggest/" + "Epoch_" + str(self.options.load_epoch)

        for i in range(n_samples):
            self.image_handler.save_image(result_imgs[uncertainties[i][1]], output_dir, save_names[uncertainties[i][1]])

    @staticmethod
    def erase_loss_file(loss_file, initial_epoch):
        """
        Function to erase all losses of future epochs
        Necessary for continued train with intermediate epoch or restart training with same name
        :param loss_file: file the losses are stored in
        :param initial_epoch: epoch to start training
        :return: None
        """

        new_content = []

        with open(loss_file, 'r') as f:
            content = f.readlines()

            for line in content:
                header, loss_data = line.split(")", maxsplit=1)

                header_value_paires = header.split(",")
                epoch = int(header_value_paires[0].split(":")[1])
                if epoch < initial_epoch:
                    new_content.append(line)

        with open(loss_file, 'w') as f:
            for line in new_content:
                f.write(line)

    def plot_losses(self, loss_file):
        """
        Function to plot loss values
        :param loss_file: file to read loss values from
        :return: None
        """
        if not os.path.isfile(loss_file):
            raise ValueError('%s is not a file' % str(loss_file))

        seg_losses = []
        epochs = []
        steps = []

        with open(loss_file, 'r') as f:
            content = f.readlines()

        content = [x.strip(" ").strip("\n") for x in content]

        for line in content:
            header, loss_data = line.split(")", maxsplit=1)

            header_value_paires = header.split(",")
            epoch = int(header_value_paires[0].split(":")[1])
            step = int(header_value_paires[1].split(":")[1])

            step_total = (epoch-1)*self.options.steps_per_epoch + step

            _tmp = str(loss_data).split(",")
            seg = _tmp[0]

            seg_losses.append(float(seg.split(":")[1]))
            epochs.append(epoch)
            steps.append(step_total)

        markers = {0: "o",
                   1: "s",
                   2: "^",
                   3: "D",
                   4: "*",
                   5: "x"
                   }

        colors = {0: "b",
                  1: "g",
                  2: "r",
                  3: "c",
                  4: "m",
                  5: "k",
                  6: "y"
                  }

        print("plotting Errors and save files to ", self.loss_dir)
        fig_losses_steps = plt.figure(1, figsize=(48, 27))
        fig_losses_epochs = plt.figure(2, figsize=(48, 27))

        figures = [fig_losses_steps, fig_losses_epochs]
        loss_labels = []
        for key, _ in self.model.get_current_errors().items():
            loss_labels.append("Loss " + str(key))
        # loss_labels = ["Loss Seg"]

        loss_list = [seg_losses]

        time_list = [steps]

        time_labels = ["Total Steps",
                       "Epochs"]

        save_paths = [self.loss_dir + "/loss_plot_steps.png",
                      self.loss_dir + "/loss_plot_epochs.png"]

        max_epoch = max(epochs)

        for j in range(len(time_list)):
            plt.figure(j + 1)
            for i, loss in enumerate(loss_list):
                ax = figures[j].add_subplot(len(loss_list), 1, i+1)
                style = markers[i % 6] + colors[i % 7] + "-"
                ax.plot(time_list[j], loss_list[i], style, label=loss_labels[i], markersize=3)
                ax.set_title(loss_labels[i])
                ax.set_xlabel(time_labels[j])
                ax.set_ylabel("Loss Values")
                if j == 0:
                    for ep in range(1, max_epoch + 1):
                        ax.axvline(ep * self.options.steps_per_epoch)

            figures[j].subplots_adjust(hspace=1.0)
            figures[j].savefig(save_paths[j])

    def create_html_file(self, current_epoch, width=400):
        """
        Function to create HTML file for better visualization
        :param current_epoch: current epoch (epoch shown at top of the HTML file)
        :param width: width of displayed images
        :return: None
        """
        print("Create HTML File")
        epoch_freq = self.options.save_img_freq

        web_dir = self.options.output_path + "/" + self.options.name + "/Train"
        self.image_handler.create_html_file(web_dir, "OverviewEpochs", "./images",
                                            current_epoch, epoch_freq, self.options.n_networks, width)
