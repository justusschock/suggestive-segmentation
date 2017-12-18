
import torch

from blocks import UnetSkipConnectionBlock


class uNet(torch.nn.Module):
    """class containing a generic generator implementation"""
    def __init__(self, n_input_channels=3, n_output_channels=1, n_blocks=9, initial_filters=64, dropout_value=0.25,
                 gpu_ids=[]
                 ):
        """
        function to create and initialize the class variables
        :param n_input_channels: number of input channels
        :param n_output_channels: number of output channels
        :param initial_filters: number of filters for first layer
        :param dropout_value: dropout value (0 for no dropout)
        :param gpu_ids: list of gpu ids
        :param block_specific_args: arguments needed to initialize blocks
        """
        super(uNet, self).__init__()
        self.input_nc = n_input_channels
        self.output_nc = n_output_channels
        self.initial_filters = initial_filters
        self.gpu_ids = gpu_ids
        self.dropout_value = dropout_value
        self.model = torch.nn.Sequential()
        self.n_blocks = n_blocks

        self._build()
        if len(gpu_ids):
            # self.model.cuda(device_id=gpu_ids[0])
            self.model.cuda()

    def _build(self):
        """
        Build model with Unet Skip Connection Block
        :return: None
        """

        nf_mult = min(8, 2 ** self.n_blocks)
        model = UnetSkipConnectionBlock(nf_mult*self.initial_filters, self.dropout_value, nf_mult*self.initial_filters, innermost=True)
        for i in range(self.n_blocks-1, 1, -1):
            nf_mult_out = min(8, 2**(int(i-2)))
            nf_mult_in = 8 if nf_mult_out == 8 else 2*nf_mult_out
            model = UnetSkipConnectionBlock(nf_mult_out*self.initial_filters, self.dropout_value, nf_mult_in*self.initial_filters, model)
        model = [UnetSkipConnectionBlock(self.input_nc, self.dropout_value, self.initial_filters, model, outermost=True)]
        model += [torch.nn.Conv2d(self.input_nc, self.output_nc, kernel_size=1), torch.nn.Sigmoid()]
        # model += [torch.nn.Conv2d(self.input_nc, self.output_nc, kernel_size=1)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input_tensor):
        """
        Function to forward through model (necessary for implicit back-propagation
        :param input_tensor: input tensor
        :return: forwarded input
        """

        if self.gpu_ids and isinstance(input_tensor.data, torch.cuda.FloatTensor):
            tmp = torch.nn.parallel.data_parallel(self.model, input_tensor, self.gpu_ids)
            # return torch.nn.parallel.data_parallel(self.model, input_tensor)
        else:
            tmp = self.model(input_tensor)

        return tmp