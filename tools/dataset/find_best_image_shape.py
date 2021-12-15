import os
import torch
from tqdm import trange

from datasets.DualCervix import DualCervixDataSet


def main():
    dataset = DualCervixDataSet(
        pipeline = [
            dict(type = 'LoadImageFromFile'),
        ],
        data_root = 'data/DualCervixDetection',
        ann_file = os.path.join('cropped_annos', 'train_{part}.json'),
        img_prefix = 'cropped_img')

    ori_shape = {}
    for i in trange(len(dataset)):
        res = dataset[i]
        for part in res:
            if part not in ori_shape:
                ori_shape[part] = {}
            if res[part]['ori_shape'] not in ori_shape[part]:
                ori_shape[part][res[part]['ori_shape']] = 1
            else:
                ori_shape[part][res[part]['ori_shape']] += 1
    print(f'ori_shape: {ori_shape}')


if __name__ == '__main__':
    main()
