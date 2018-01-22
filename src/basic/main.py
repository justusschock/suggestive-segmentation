import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib
matplotlib.use('Agg')
from PIL import Image
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

    # img1 = Image.open("/home/temp/schock/MGS/Eyes/Images/W2_MGS_MVI_0297_4.png")
    # img2 = Image.open("/home/temp/schock/MGS/Eyes/Images/W2_MGS_MVI_0297_18.png")
    # imgs = [img1, img2]
    # preds = model.predict(imgs)
    model.train()
    model.predict_to_dir(0)
