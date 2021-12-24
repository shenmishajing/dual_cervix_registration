import os
import os.path as osp
import numpy as np

import torch

from typing import Optional, Mapping, Any

from .coco import CocoDataset
from .api_wrappers import COCO
from .pipelines import Compose
from .base import LightningDataModule


class DualCervixDataSet(CocoDataset):
    CLASSES = ('hsil',)
    Modals = ['acid', 'iodine']

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes = None,
                 data_root = None,
                 img_prefix = '',
                 seg_prefix = None,
                 proposal_file = None,
                 test_mode = False,
                 filter_empty_gt = True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = {part: [self.data_infos[part][i] for i in valid_inds] for part in self.Modals}
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        if not isinstance(pipeline, dict):
            pipeline = {part: pipeline for part in self.Modals}
        self.pipeline = {part: Compose(pipeline[part]) for part in self.Modals}

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos[self.Modals[0]])

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            dict[str, list[dict]]: Annotation info from COCO api.
        """

        self.coco = {part: COCO(ann_file.format(part = part)) for part in self.Modals}
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco[self.Modals[0]].get_cat_ids(cat_names = self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco[self.Modals[0]].get_img_ids()
        data_infos = {}
        for part in self.Modals:
            data_infos[part] = []
            total_ann_ids = []
            for i in self.img_ids:
                info = self.coco[part].load_imgs([i])[0]
                info['filename'] = info['file_name']
                data_infos[part].append(info)
                ann_ids = self.coco[part].get_ann_ids(img_ids = [i])
                total_ann_ids.extend(ann_ids)
            assert len(set(total_ann_ids)) == len(
                total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx, part = None):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if part is None:
            img_id = self.data_infos[self.Modals[0]][idx]['id']
            ann_ids = {part: self.coco[part].get_ann_ids(img_ids = [img_id]) for part in self.Modals}
            ann_info = {part: self.coco[part].load_anns(ann_ids[part]) for part in self.Modals}
            ann_info = {part: self._parse_ann_info(self.data_infos[part][idx], ann_info[part]) for part in self.Modals}
        else:
            img_id = self.data_infos[self.Modals[0]][idx]['id']
            ann_ids = self.coco[part].get_ann_ids(img_ids = [img_id])
            ann_info = self.coco[part].load_anns(ann_ids)
            ann_info = self._parse_ann_info(self.data_infos[part][idx], ann_info)
        return ann_info

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[self.Modals[0]][idx]['id']
        ann_ids = {part: self.coco[part].get_ann_ids(img_ids = [img_id]) for part in self.Modals}
        ann_info = {part: [ann['category_id'] for ann in self.coco[part].load_anns(ann_ids[part])] for part in self.Modals}
        return ann_info

    def _filter_imgs(self, min_size = 32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco[self.Modals[0]].anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco[self.Modals[0]].cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos[self.Modals[0]]):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype = np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[self.Modals[0]][i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        res = {}
        random_state = np.random.get_state()
        for part in self.Modals:
            np.random.set_state(random_state)
            img_info = self.data_infos[part][idx]
            ann_info = self.get_ann_info(idx, part)
            results = dict(img_info = img_info, ann_info = ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            res[part] = self.pipeline[part](results)
        return res

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        res = {}
        random_state = np.random.get_state()
        for part in self.Modals:
            np.random.set_state(random_state)
            img_info = self.data_infos[part][idx]
            results = dict(img_info = img_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            res[part] = self.pipeline[part](results)
        return res


class DualCervixDataModule(LightningDataModule):
    def __init__(self,
                 ann_path: str,
                 train_pipeline: Mapping[str, Any],
                 test_pipeline: Mapping[str, Any],
                 data_root: Optional[str] = '.',
                 img_prefix: Optional[str] = '',
                 seg_prefix: Optional[str] = '',
                 data_loader_config: Optional[Mapping[str, Any]] = None):
        super().__init__(data_loader_config)
        self.ann_path = ann_path
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix

    def _build_data_set(self, split):
        return DualCervixDataSet(ann_file = os.path.join(self.ann_path, split + '_{part}.json'),
                                 pipeline = self.train_pipeline,
                                 # pipeline = self.train_pipeline if split == 'train' else self.test_pipeline,
                                 data_root = self.data_root,
                                 img_prefix = self.img_prefix,
                                 seg_prefix = self.seg_prefix)

    @staticmethod
    def collate_fn(batch):
        res = {}
        for part in batch[0]:
            res[part] = {}
            stack_keys = ['img', 'gt_segments_from_bboxes']
            for key in [k for k in stack_keys if k in batch[0][part]]:
                if isinstance(batch[0][part][key], torch.Tensor):
                    res[part][key] = torch.stack([x[part][key] for x in batch])
                else:
                    res[part][key] = [x[part][key] for x in batch]
            for key in [k for k in batch[0][part] if k not in stack_keys]:
                res[part][key] = [x[part][key] for x in batch]
        return res
