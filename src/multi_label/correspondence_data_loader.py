import torch.utils.data
import torchvision.transforms as transforms
from image_folder import CorrespondenceImageFolder, get_corresponding_path, \
    default_loader, default_label_loader, \
    CorrespondenceImageFolderMultiLabel, default_label_loader_multi, get_corresponding_path_multi, ImageFolder


class CorrespondenceDataLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.data_set = None
        self.data_loader = None

        self.initialize(opt)

    def initialize(self, opt):

        # Define Transformations
        if opt.input_nc == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opt.input_nc == 1:
            norm = transforms.Normalize([0], [1])
        else:
            raise(RuntimeError('input_nc not supported'))

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            norm])

        if opt.output_nc == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opt.output_nc == 1:
            norm = transforms.Normalize([0], [1])
        else:
            raise(RuntimeError('input_nc not supported'))

        output_transform = transforms.Compose([
            transforms.ToTensor(),
            norm])

        if opt.phase_test:
            root = [opt.test_path]
        else:

            root = opt.dataroot

        self.data_set, self.data_loader = [], []

        for path in root:

            # Create Data Set
            dataset = CorrespondenceImageFolder(root=path,
                                                input_nc=opt.input_nc,
                                                output_nc=opt.output_nc,
                                                height=opt.image_height,
                                                width=opt.image_width,
                                                input_transform=input_transform, output_transform=output_transform,
                                                return_paths=True)
            self.data_set.append(dataset)
            # Create Data Loader
            self.data_loader.append(torch.utils.data.DataLoader(dataset,
                                                                batch_size=opt.batch_size,
                                                                shuffle=opt.shuffle,
                                                                num_workers=0))

    def name(self):
        return 'CorrespondenceDataLoader'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return [len(dataset) for dataset in self.data_set]


class CorrespondenceDataLoaderMultiLabel(object):
    def __init__(self, opt):
        self.opt = opt
        self.data_set = None
        self.data_loader = None

        self.initialize(opt)

    def initialize(self, opt):

        # Define Transformations
        if opt.input_nc == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opt.input_nc == 1:
            norm = transforms.Normalize([0], [1])
        else:
            raise(RuntimeError('input_nc not supported'))

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            norm])

        if opt.output_nc == 3:
            norm = transforms.Normalize([0.5 for x in range(opt.n_labels)], (0.5 for x in range(opt.n_labels)))
        elif opt.output_nc == 1:
            norm = transforms.Normalize([0 for x in range(opt.n_labels)], (1 for x in range(opt.n_labels)))
        else:
            raise (RuntimeError('output_nc not supported'))

        output_transform = transforms.Compose([
            transforms.ToTensor(),
            norm])

        if opt.phase_test:
            root = [opt.test_path]
        else:

            root = opt.dataroot

        self.data_set, self.data_loader = [], []

        img_loader = default_loader
        label_loader = default_label_loader_multi
        path_mapper = get_corresponding_path_multi

        for path in root:

            # Create Data Set
            dataset = CorrespondenceImageFolderMultiLabel(root=path,
                                                          input_nc=opt.input_nc,
                                                          output_nc=opt.output_nc,
                                                          height=opt.image_height,
                                                          width=opt.image_width,
                                                          input_transform=input_transform,
                                                          output_transform=output_transform,
                                                          return_paths=True,
                                                          n_labels=opt.n_labels,
                                                          loader=img_loader,
                                                          label_loader=label_loader,
                                                          path_mapper=path_mapper,
                                                          return_labels=not opt.phase_test)
            self.data_set.append(dataset)
            # Create Data Loader
            self.data_loader.append(torch.utils.data.DataLoader(dataset,
                                                                batch_size=opt.batch_size,
                                                                shuffle=opt.shuffle,
                                                                num_workers=0))

    def name(self):
        return 'CorrespondenceDataLoaderMultiLabel'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return [len(dataset) for dataset in self.data_set]
