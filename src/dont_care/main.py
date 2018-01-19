import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import matplotlib
matplotlib.use('Agg')
from net_config import NetConfig
from segmentation_nn import SegmentationNetwork


if __name__ == '__main__':

    base_path = ""

    if sys.platform == 'win32':
        base_path = str(os.path.abspath(__file__)).replace("\\main.py", "")
    else:
        base_path = str(os.path.abspath(__file__)).replace("/main.py", "")

    options = NetConfig('seg_config.yaml')
    gpu_ids = [0]

    model = SegmentationNetwork(options=options, gpu_ids=gpu_ids)

    if options.phase_test:
        model.predict_to_dir(n_predicts=0)
    else:
        model.train()
        model.predict_to_dir(0)
