
import torch.utils.data as data

from PIL import Image, ImageOps
import os
import os.path

import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif', '.GIF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    returns paths of valid images in dir, sorted
    :param dir:
    :return:
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and not \
                    ("_mask" in fname.rsplit(".")[0] or "_dont_care" in fname.rsplit(".")[0]):
                path = os.path.join(root, fname)
                images.append(path)

    images.sort()

    return images


def get_corresponding_path(img_path, label_root_path):
    img_name = (os.path.split(img_path)[-1]).rsplit('.', 1)[0]

    label_name = img_name + "_mask.png"

    return os.path.join(label_root_path, label_name)


def get_corresponding_path_multi(img_path, label_root_path):
    return get_corresponding_path(img_path, label_root_path)


def default_loader(path, n_channels, height, width):
    img = Image.open(path)

    if n_channels == 1:
        img = img.convert('L')

    else:
        img = img.convert('RGB')

    return img.resize((width, height), Image.BILINEAR)


def default_label_loader_multi(path, n_channels, height, width, n_labels):
    labels = []

    for label_idx in range(1, n_labels+1):
        labels.append(np.array(default_label_loader(path.replace('.png', '_%d.png' % label_idx),
                                                    n_channels, height, width)))

    mask = np.stack(labels, axis=-1)
    return mask


def default_label_loader(path, n_channels, height, width):
    return default_loader(path, n_channels, height, width)


def car_loader(path, n_channels):
    img = Image.open(path)

    if n_channels == 1:
        img = img.convert('L')

    else:
        img = img.convert('RGB')

    img = ImageOps.expand(img, border=(1, 0), fill='black')

    return img


class ImageFolder(data.Dataset):

    def __init__(self, root, n_channels, transform=None, return_paths=False,
                 loader=car_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                                              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.n_channels = n_channels

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path, self.n_channels)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class CorrespondenceImageFolder(data.Dataset):

    def __init__(self, root, input_nc, output_nc, height=512, width=512, input_transform=None, output_transform=None, return_labels=True, return_paths=False,
                 loader=default_loader, label_loader=default_label_loader, path_mapper=get_corresponding_path):
        img_names = make_dataset(root)
        if len(img_names) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                                              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.img_names = img_names
        self.return_labels = return_labels
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.return_paths = return_paths
        self.loader = loader
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.label_loader = label_loader
        self.height = height
        self.width = width
        self.path_mapper = path_mapper

    def __getitem__(self, index):

        path = self.img_names[index]
        img_pil = self.loader(path, self.input_nc, self.height, self.width)
        if self.input_transform is not None:
            img = self.input_transform(img_pil)
        else:
            img = img_pil

        if self.return_labels:
            path_label = self.path_mapper(path, self.root)
            label_pil = self.label_loader(path_label, self.output_nc, self.height, self.width)
            if self.output_transform is not None:
                label = self.output_transform(label_pil)
            else:
                label = label_pil

            if self.return_paths:
                data_dict = {'img': img, 'label': label, 'path_img': path, 'path_label': path_label}
            else:
                data_dict = {'img': img, 'label': label}

        else:
            if self.return_paths:
                data_dict = {'img': img, 'path_img': path}
            else:
                data_dict = {'img': img}

        return data_dict

    def __len__(self):
        return len(self.img_names)


class CorrespondenceImageFolderMultiLabel(data.Dataset):

    def __init__(self, root, input_nc, output_nc, height=512, width=512, input_transform=None, output_transform=None,
                 return_labels=True, return_paths=False, n_labels=3, loader=default_loader,
                 label_loader=default_label_loader_multi, path_mapper=get_corresponding_path_multi):
        img_names = make_dataset(root)
        if len(img_names) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                                                              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.img_names = img_names
        self.return_labels = return_labels
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.return_paths = return_paths
        self.loader = loader
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.label_loader = label_loader
        self.height = height
        self.width = width
        self.path_mapper = path_mapper
        self.n_labels = n_labels

    def __getitem__(self, index):

        path = self.img_names[index]
        img_pil = self.loader(path, self.input_nc, self.height, self.width)
        if self.input_transform is not None:
            img = self.input_transform(img_pil)
        else:
            img = img_pil

        data_dict = {'img': img}

        if self.return_labels:
            path_label = self.path_mapper(path, self.root)
            label_pil = self.label_loader(path_label, self.output_nc, self.height, self.width, self.n_labels)

            if self.output_transform is not None:
                label = self.output_transform(label_pil)

            else:
                label = label_pil

            data_dict['label'] = label

        if self.return_paths:
            data_dict['path_img'] = path

            if self.return_labels:
                data_dict['path_label'] = path_label

        return data_dict

    def __len__(self):
        return len(self.img_names)

