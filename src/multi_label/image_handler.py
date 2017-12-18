import os
import numpy as np
from yattag import Doc
from yattag import indent
from PIL import Image


class ImageHandler(object):
    """Class for handling image output"""

    class HtmlViewer(object):
        """class for creating html files to visualize outputs"""
        def __init__(self, title):
            """
            function to create and initialize class variables
            :param title: document title
            """
            doc, tag, text = Doc(
                defaults={'title': title}
            ).tagtext()
            self.doc = doc
            self.tag = tag
            self.text = text

        def add_header(self, header_str):
            """
            function to add heading line in document
            :param header_str:
            :return: None
            """
            with self.tag('h3'):
                self.text(header_str)

        def add_table(self, border=1):
            """
            function to add table to document
            :param border: determination whether to show borders or not
            :return: HTML table tag
            """
            t = self.tag('table', ('border', border), ('style', 'table-layout: fixed;'))
            return t

        def add_images(self, imgs, txts, links, width=400):
            """
            Function to add images to document
            :param imgs: list of images
            :param txts: list of texts corresponding to images
            :param links: lists of links to images
            :param width: image width
            :return: None
            """
            with self.add_table():
                with self.tag('tr'):
                    for img, txt, link in zip(imgs, txts, links):
                        with self.tag('td',
                                      ('style', 'word-wrap: break-word;'),
                                      ('halign', 'center'),
                                      ('valign', 'top')):
                            with self.tag('p'):
                                with self.tag('a', ('href', link)):
                                    self.doc.stag('img', style="width:%dpx" % width,
                                                  src=img)

                                    self.tag('br')

                                    with self.tag('p'):
                                        self.text(txt)

        def save(self, filename, file_dir):
            """
            function to save HTML-File
            :param filename: name of HTML File (including '.html' extension)
            :param file_dir: directory the file should be saved in
            :return: None
            """
            html_file = os.path.join(file_dir, filename)

            with open(html_file, 'wt') as f:
                f.write(indent(self.doc.getvalue()))

    def __init__(self):
        """function to create and initialize class variables"""
        self.html_viewer = None

    def create_html_file(self, web_dir, title, image_dir, epochs, epoch_freq, n_models, width):
        """
        Function to create a html File
        :param web_dir: directory the html file should be saved in
        :param title: title of the html file
        :param image_dir: directory containing the images which should be displayed in the html file
        :param epochs: number of passed epochs
        :param epoch_freq: frequency of saving images
        :param width: width of displayed images
        :param saverec: True if reconstructions have been saved, False otherwise
        :return: None
        """
        self.html_viewer = self.HtmlViewer(title)

        for n in range(epochs, 0, -1):

            images = []
            labels = []
            links = []

            if n % epoch_freq == 0:
                self.html_viewer.add_header('Epoch %d' % n)

                for model_idx in range(n_models):

                    label_real_img = "Real Img Model %d" % model_idx
                    label_fake_mask = "Fake Mask Model %d" % model_idx
                    label_gt = "GT Model %d" % model_idx

                    img_path_real_img = image_dir + "/epoch_%03d_real_img_model_%d.png" % (n, model_idx)
                    img_path_fake_mask = image_dir + "/epoch_%03d_fake_mask_model_%d.png" % (n, model_idx)
                    img_path_gt = image_dir + "/epoch_%03d_gt_model_%d.png" % (n, model_idx)

                    images.append(img_path_real_img)
                    labels.append(label_real_img)
                    links.append(img_path_real_img)

                    images.append(img_path_fake_mask)
                    labels.append(label_fake_mask)
                    links.append(img_path_fake_mask)

                    images.append(img_path_gt)
                    labels.append(label_gt)
                    links.append(img_path_gt)

            self.html_viewer.add_images(images, labels, links, width=width)
        self.html_viewer.save(title + ".html", web_dir)

    @staticmethod
    def _tensor_to_image(tensor, mask=False):
        """
        Converting pytorch-tensor to numpy-array
        :param tensor: pytorch tensor
        :return: img: numpy array with tensor data
        """
        numpy_image = tensor.data[0].cpu().float().numpy()
        if mask:
            img = (np.transpose(numpy_image, (1, 2, 0))) * 255.0
        else:
            img = (np.transpose(numpy_image, (1, 2, 0)) + 1) / 2 * 255.0
        return img

    @staticmethod
    def save_image(image, image_path, name, extension='.png'):
        """
        function to save images after getting the image_data from tensors
        :param image: tensor containing the image
        :param image_path: path the image should be saved to
        :param name: file name
        :param extension: file extension
        :return: None
        """
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        extension = str(extension) if str(extension).startswith('.') else '.' + str(extension)
        save_path = image_path + "/" + name + extension

        img = ImageHandler._tensor_to_image(image)
        img = img.astype(np.uint8)

        img = Image.fromarray(np.squeeze(img)).convert('RGB')

        img.save(save_path)

    @staticmethod
    def save_mask(image, image_path, name, extension='.png', threshold=0.9):
        """
        function to save images after getting the image_data from tensors
        :param image: tensor containing the image
        :param image_path: path the image should be saved to
        :param name: file name
        :param extension: file extension
        :return: None
        """
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        extension = str(extension) if str(extension).startswith('.') else '.' + str(extension)
        save_path = image_path + "/" + name + extension

        img = ImageHandler._tensor_to_image(image, mask=True)
        img = img.astype(np.uint8)

        img = Image.fromarray(np.squeeze(img)).convert('RGB')

        img.save(save_path)

    @staticmethod
    def save_image_default(image, image_path, name, extension='.png'):
        """
        function to save images after getting the image_data from tensors
        :param image: tensor containing the image
        :param image_path: path the image should be saved to
        :param name: file name
        :param extension: file extension
        :return: None
        """
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        extension = str(extension) if str(extension).startswith('.') else '.' + str(extension)
        save_path = image_path + "/" + name + extension

        img = ImageHandler._tensor_to_image(image)
        img = img.astype(np.uint8)

        img = Image.fromarray(np.squeeze(img)).convert('RGB')

        img.save(save_path)
