import torch.utils.data
import torchvision.transforms as transforms
from image_folder import ImageFolder


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, shuffle=True):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.shuffle = shuffle

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        return self

    def __next__(self):
        # if self.shuffle:
        #     n_iter = randint(1, min(len(self.data_loader_A), len(self.data_loader_B)))
        # else:
        #     n_iter = 1

        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)
        try:
            B, B_paths = next(self.data_loader_B_iter)

        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if self.stop_A and self.stop_B:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            return {'A': A, 'A_paths': A_paths,
                    'B': B, 'B_paths': B_paths}


class UnalignedDataLoader(object):
    def __init__(self, opt):
        self.opt = opt

        self.initialize(opt)

    def initialize(self, opt):
        if opt.image_width > opt.image_height:
            scale_size = opt.image_height
        else:
            scale_size = opt.image_width


        # Dataset A
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            norm])
        dataset_A = ImageFolder(root=opt.dataroot + '/A', n_channels=opt.input_nc,
                                transform=transform, return_paths=True)
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=0)

        # Dataset B
        norm = transforms.Normalize([0.5], [0.5])
        transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            norm])
        dataset_B = ImageFolder(root=opt.dataroot + '/B', n_channels=opt.output_nc,
                                transform=transform, return_paths=True)
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=0)
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.paired_data = PairedData(data_loader_A, data_loader_B)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(len(self.dataset_A), len(self.dataset_B))


