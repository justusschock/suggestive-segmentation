import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import shutil
from multiprocessing import Pool, freeze_support
from functools import partial


def create_npy_label(file_name, in_path, out_path):
    img = np.array(Image.open(os.path.join(in_path, file_name)).convert('L'))
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.close()

    background = np.zeros_like(img)
    background[0:150, :] = 1
    background[-150:-1, :] = 1

    # plt.imshow(background, cmap='gray')
    # plt.show()
    # plt.close()

    or_img = np.logical_or(background == 1, img == 255)
    # plt.imshow(or_img, cmap='gray')
    # plt.show()
    # plt.close()

    dont_care_mask = np.logical_not(or_img).astype(np.uint8)*255
    # plt.imshow(dont_care_mask, cmap='gray')
    # plt.show()
    # plt.close()
    #
    # out = np.empty((2, *img.shape))
    # out[0, :, :] = dont_care_mask
    # out[1, :, :] = img

    dont_care_mask = Image.fromarray(dont_care_mask)
    dont_care_mask.save(os.path.join(out_path, file_name.replace("_mask.png", "_dont_care.png")))
    # np.save(os.path.join(out_path, file_name.replace("_mask.png", "_dont_care.png")), out)
    shutil.copy2(os.path.join(in_path, file_name.replace("_mask.png", ".jpg")), os.path.join(out_path, file_name.replace("_mask.png", ".jpg")))
    shutil.copy2(os.path.join(in_path, file_name), os.path.join(out_path, file_name))


def main():
    in_path = '/home/temp/schock/MGS/Test/whole'
    out_path = '/home/temp/schock/MGS/Test/whole_dont_care'

    os.makedirs(out_path, exist_ok=True)

    file_list = [x for x in os.listdir(in_path) if x.endswith('_mask.png')]

    with Pool() as p:
        p.map(partial(create_npy_label, in_path=in_path, out_path=out_path), file_list)


if __name__ == '__main__':
    freeze_support()
    main()


