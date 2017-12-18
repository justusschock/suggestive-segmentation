import random
import numpy as np
import torch


class ImagePool:
    """class to buffer images"""
    def __init__(self, poolsize=50):
        """
        function to create and initialize class variables
        :param poolsize: number of buffered images
        """
        self.pool_size = poolsize

        if self.pool_size:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        function to handle the buffering (50 percent probability tha a image is buffered)
        :param images: list of pytorch tensors containing the images
        :return: pytorch variable containing the images
        """
        if not self.pool_size:
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # 50 percent probability to replace random image in self.images with currently considered image
                    #  and return replaced image
                    random_id = random.randint(0, self.pool_size-1)
                    #tmp = np.copy(self.images[random_id])
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # 50 percent probability to return currently considered image
                    return_images.append(image)
        temp = torch.cat(return_images, 0)
        return torch.autograd.Variable(temp.data)

    def get_random_item(self, batch_size):
        return_images = []
        for i in range(batch_size):
            random_id = random.randint(0, self.pool_size - 1)
            tmp = self.images[random_id].clone()
            return_images.append(tmp)
        temp = torch.cat(return_images, 0)
        return torch.autograd.Variable(temp.data)