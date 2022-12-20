import os
import pathlib
import sys
from os.path import isfile, join

import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from classes.ts_detection import TorchscriptDetection
from utils.util import cv2_imshow


def get_mask(image_mask, box):
    """
    Get real mask from 28x28 mask
    Parameters:
    ----------
        image_mask:  np.array -  image mask (black-white)
    Returns:
    -------
        new_mask: np.array
    """

    width_box = int(box[2]) #- int(box[0])
    height_box = int(box[3]) #- int(box[1])
    # print(f'width_box={width_box}, height_box={height_box}')
    mask = image_mask[0]  # .astype('uint8')
    mask[mask > 0.9] = 1.0
    mask = mask.astype('uint8')
    new_mask = cv2.resize(mask, (width_box, height_box))
    return new_mask

def get_image(image, bbox, mask) -> np.ndarray:
    """
    Get black-white image with colored ROI
    _________
    :param img:
    :type img: np.ndarray
    :param pr_boxes:
    :type pr_boxes: list
    :param masks:
    :type masks: list
    :return:
    :rtype: np.ndarray
    """
    bbox = [int(x) for x in bbox]
    x0, y0, w, h = bbox
    img = image[y0:y0 + h, x0:x0 + w]
    # img_blank = np.ones_like(img) * 255
    image_mask = get_mask(mask, bbox) * 255
    neg_mask = cv2.bitwise_not(image_mask)
    neg_masks = cv2.merge((neg_mask, neg_mask, neg_mask))
    # cv2_imshow(neg_mask)

    image2 = cv2.bitwise_and(img, img, mask=image_mask)
    image2 = cv2.bitwise_or(image2, neg_masks) #
    # image2 = cv2.merge((img, image_mask))
    # print(img.shape)
    # image2 =np.expand_dims(img, axis=3)
    # print(image2.shape)
    # image2[:, :, :-1] = image_mask
    # image2 = np.concatenate((img, image_mask), axis=2)

    return image2

def main(args):
    # print(f'args={args}')
    input_dir = args.input_dir
    # output_dir = args.output_dir

    tsd = TorchscriptDetection('../weights/potato_best2022121821_192_1cl.ts', use_cuda=False)
    # path = r'C:\softz\work\potato\in\images\from_video\potatos6'
    # file = '106.jpg'
    if args.output_dir is None:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f)) and
             join(input_dir, f).split('.')[1] == 'jpg']
    count = 0
    for file in tqdm(files):
        file_name, ext = file.split('.')
        image = cv2.imread(os.path.join(input_dir, file))
        pred_boxes, scores, _, masks = tsd.detect(image)

        # print(f'scores={scores}')
        # print(f'pred_classes={pred_classes}')
        # print(f'pred_masks={pred_masks[0].shape}')
        for i, (pred_box, score, mask) in enumerate(zip(pred_boxes, scores, masks)):
            if score > 0.7:
                # print(f'pred_box={pred_box}')

                image1 = get_image(image, pred_box, mask)
                # cv2_imshow(image1, 'image1')
                cv2.imwrite(os.path.join(output_dir,f'{file_name}_{str(i)}.jpg'), image1)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate png files")
    parser.add_argument(
        "-id", "--input_dir",
        help="input path to a JPG files and output path by default "
    )
    parser.add_argument(
        "-od", "--output_dir",  default=None,
        help="another path to a resized (changed)  PNG files"
    )
    main(parser.parse_args())