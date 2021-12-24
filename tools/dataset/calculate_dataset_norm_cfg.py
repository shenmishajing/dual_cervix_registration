import os
import torch
from tqdm import trange

from datasets.DualCervix import DualCervixDataSet


def add_item(sum_dict, part, item, index):
    if part not in sum_dict:
        sum_dict[part] = item
    else:
        sum_dict[part] = sum_dict[part] * (index / (index + 1)) + item / (index + 1)


def main():
    device = 'cpu'
    dataset = DualCervixDataSet(
        pipeline = [
            dict(type = 'LoadImageFromFile',
                 color_type = 'grayscale'),
            dict(type = 'Resize', img_scale = (608, 608), keep_ratio = False),
        ],
        data_root = 'data/DualCervixDetection',
        ann_file = os.path.join('cropped_annos', 'train_{part}.json'),
        img_prefix = 'cropped_img')

    mean, std = [{} for _ in range(2)]
    with torch.no_grad():
        for i in trange(len(dataset)):
            res = dataset[i]
            res = {part: torch.mean(torch.tensor(v['img'], dtype = torch.float, device = device), dim = [0, 1]) for part, v in res.items()}
            res['total'] = torch.mean(torch.stack(list(res.values())), dim = 0)
            for part, v in res.items():
                add_item(mean, part, res[part], i)
        mean_list = {k: m.cpu().numpy().tolist() for k, m in mean.items()}
        print(f'mean: {mean_list}')

        for i in trange(len(dataset)):
            res = dataset[i]
            res = {part: torch.pow(torch.mean(torch.tensor(v['img'], dtype = torch.float, device = device), dim = [0, 1]) - mean[part], 2)
                   for part, v in res.items()}
            res['total'] = torch.mean(torch.stack(list(res.values())), dim = 0)
            for part, v in res.items():
                add_item(std, part, res[part], i)
        std_list = {k: torch.sqrt(s).cpu().numpy().tolist() for k, s in std.items()}
        print(f'std: {std_list}')


if __name__ == '__main__':
    main()
