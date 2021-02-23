#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/2/2 17:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/2 17:07   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import itertools
import json
import os
import cv2 as cv
import os.path as osp
import sys
import tempfile
import time
from copy import deepcopy
import re
import mmcv
from mmcv.utils import get_logger
from mmcv.utils import print_log
import logging
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from object_detection_server.mmdetection_api import seal_detection


class COCODatasetEvaluation():
    """
    @author:wangfc27441
    @desc:
         主要来自 mmdet/datasets/coco.py   class CocoDataset(CustomDataset)
    @version：
    @time:2021/2/3 13:56

    Parameters
    ----------

    Returns
    -------
    """
    # 自定义数据集的类型
    CLASSES = ("back_idcard", "ellipse_seal", "front_idcard", "rectangle_name_seal", "round_seal", "square_name_seal")

    def __init__(self, data_root, img_prefix='test', ann_file='annotations/test.json', only_seal_detection=True,
                 if_remove_other_supercategory=True,
                 test_mode=True, classes=None):
        # 判断是否需要转换为 印章父类型
        self.only_seal_detection = only_seal_detection
        self.if_remove_other_supercategory = if_remove_other_supercategory
        self.supercategory_annotations_json_file = 'supercategory.json'
        self.supercategory_annotations_json_path = None
        self.supercategory2new_id_dict = None
        self.name_id2supercategory_id_dict = None
        self.name2supercategory_id_dict = None

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        # self.seg_prefix = seg_prefix
        # self.proposal_file = proposal_file
        self.test_mode = test_mode
        # self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            # if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
            #     self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            # if not (self.proposal_file is None
            #         or osp.isabs(self.proposal_file)):
            #     self.proposal_file = osp.join(self.data_root,
            #                                   self.proposal_file)

        if self.only_seal_detection:
            self.get_supercategory_annotations(self.ann_file)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        # if self.proposal_file is not None:
        #     self.proposals = self.load_proposals(self.proposal_file)
        # else:
        #     self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        # self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def getitem(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        # results = dict(img_info=img_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        # self.pre_pipeline(results)
        # return self.pipeline(results)
        return img_info

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def get_supercategory_annotations(self, ann_file):
        """
        @author:wangfc27441
        @desc:
        # 将 annotation 转换为 父类的形式
        @version：
        @time:2021/2/3 17:27

        Parameters
        ----------

        Returns
        -------
        """
        with open(ann_file, mode='r') as f:
            ann_json = json.load(f)
        # ori_coco = COCO(ann_file)
        # ori_categories = ori_coco.dataset['categories']
        ori_categories = ann_json['categories']
        ori_annotations = ann_json['annotations']

        supercategories = set()
        id2name = {}
        name2id = {}
        name2supercategory = {}
        for category_info in ori_categories:
            supercategory = category_info['supercategory']
            name = category_info['name']
            id = category_info['id']
            supercategories.add(supercategory)
            id2name.update({id: name})
            name2id.update({name: id})
            name2supercategory.update({name: supercategory})

        if self.if_remove_other_supercategory:
            new_supercategories = []
            for supercategory in supercategories:
                matched = re.match(pattern='[\s\S]{0,10}seal$', string=supercategory)
                if matched:
                    new_supercategories.append(supercategory)
        else:
            new_supercategories = supercategories
        supercategories = sorted(new_supercategories)

        supercategory2new_id = {supercategory: i + 1 for i, supercategory in enumerate(supercategories)}

        def name_id2supercategory_id(name_id):
            name = id2name[name_id]
            supercategory = name2supercategory[name]
            # 如果 supercategory，则默认值为 0
            new_id = supercategory2new_id.get(supercategory, 0)
            return new_id

        def name2supercategory_id(name):
            supercategory = name2supercategory[name]
            new_id = supercategory2new_id.get(supercategory, 0)
            return new_id

        self.name_id2supercategory_id_dict = {name_id: name_id2supercategory_id(name_id) for name_id, name in
                                              id2name.items()}
        self.supercategory2new_id_dict = supercategory2new_id
        self.name2supercategory_id_dict = {name: name2supercategory_id(name) for name, id in name2id.items()}

        new_ann_json = deepcopy(ann_json)
        new_categories = [{'name': supercategory, 'id': new_id} for supercategory, new_id in
                          supercategory2new_id.items()]
        new_annotations = []
        for annotation in ori_annotations:
            category_id = annotation['category_id']
            new_id = name_id2supercategory_id(category_id)
            annotation.update({'category_id': new_id})
            new_annotations.append(annotation)
        new_ann_json.update({'categories': new_categories, 'annotations': new_annotations})

        new_ann_json_path = osp.join(osp.dirname(ann_file), self.supercategory_annotations_json_file)
        self.supercategory_annotations_json_path = new_ann_json_path
        with open(new_ann_json_path, mode='w', encoding='utf-8') as f:
            json.dump(new_ann_json, f, indent=4, ensure_ascii=False)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        if self.only_seal_detection:
            # 加载 父类的标注信息作为 annotations
            self.coco = COCO(self.supercategory_annotations_json_path)
            cat_names = sorted([name_info['name'] for name_info in self.coco.cats.values()])
            self.cat_ids = self.coco.get_cat_ids(cat_names=cat_names)

        else:
            self.coco = COCO(ann_file)
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def position2xywh(self, position):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        # _bbox = bbox.tolist()
        # return [
        #     _bbox[0],
        #     _bbox[1],
        #     _bbox[2] - _bbox[0],
        #     _bbox[3] - _bbox[1],
        # ]
        x1, y1, x2, y2, x3, y3, x4, y4 = position
        w = x3 - x1
        h = y3 - y1
        return [x1, y1, w, h]

    def _det_dict2json(self, results):
        """Convert detection results to COCO json style.
        将 api 识别的结果转换为 COCO son style
        result = {'image_name':  , 'prediction_result': }
        """
        json_results = []
        for idx in range(len(self.img_ids)):
            # 按顺序得到 img_id 和 result
            img_id = self.img_ids[idx]
            data_info = self.data_infos[idx]
            filename_from_ann = data_info['file_name']
            result = results[idx]
            filename = result['filename']
            # 确保标注的图片和测试的图片一致
            assert filename_from_ann == filename
            bbox_results = result['bbox_results']

            for i in range(bbox_results.__len__()):
                bbox_result = bbox_results[i]
                position = bbox_result['position']
                label = bbox_result["label"]
                probability = bbox_result['probability']

                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.position2xywh(position)
                data['score'] = float(probability)
                if self.only_seal_detection:
                    # 使用 输出的 label 来返回 supercategory_id
                    category_id = self.supercategory2new_id_dict[label]
                else:
                    category_id = self.cat_ids[label]

                data['category_id'] = category_id
                json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        # 增加 识别结果为 dict 的情况
        elif isinstance(results[0], dict):
            json_results = self._det_dict2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == self.img_ids.__len__(), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results),  self.img_ids.__len__()))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # 将 result 转换为 coco json 格式
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # 标注的信息
        cocoGt = self.coco

        eval_results = {}

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            # if metric == 'proposal_fast':
            #     ar = self.fast_eval_recall(
            #         results, proposal_nums, iou_thrs, logger='silent')
            #     log_msg = []
            #     for i, num in enumerate(proposal_nums):
            #         eval_results[f'AR@{num}'] = ar[i]
            #         log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
            #     log_msg = ''.join(log_msg)
            #     print_log(log_msg, logger=logger)
            #     continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def evaluate_single_result(self,
                               image_index,
                               result,
                               metric='bbox',
                               logger=None,
                               jsonfile_prefix=None,
                               classwise=False,
                               proposal_nums=(100),
                               iou_thrs=None,
                               metric_items=None):
        """
        进行单个结果的评估
        Evaluation in COCO protocol.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # 将 result 转换为 coco json 格式
        result_files, tmp_dir = self.format_results([result], jsonfile_prefix)
        # 标注的信息
        cocoGt = self.coco

        eval_results = {}

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


def detection_evaluate(model, data_root, img_prefix='test', ann_file='annotations/test.json'):
    coco_dataset_evaluation = COCODatasetEvaluation(
        data_root=data_root,
        img_prefix=img_prefix,
        ann_file=ann_file)

    image_size = coco_dataset_evaluation.img_ids.__len__()
    results = []
    start_time = time.time()
    for image_idx, image_id in enumerate(coco_dataset_evaluation.img_ids):
        # 获取每张图片的信息
        img_info = coco_dataset_evaluation.getitem(idx=image_idx)
        filename = img_info['filename']
        image_id_in_annotation = img_info['id']
        # 验证  image_idx  输出的图片 image_id 和标注的数据一致
        assert image_id == image_id_in_annotation
        test_img_path = os.path.join(data_root, img_prefix, filename)
        print(f'image_idx={image_idx},image_id={image_id},test_img_path={test_img_path}')
        test_img_bgr = cv.imread(test_img_path, 1)  # load image as bgr
        # 使用模型进行预测
        bbox_results, processed_image = seal_detection(model=model, image=test_img_bgr)
        print(f'bbox_results={bbox_results}')
        # img_drawed = show_result(img_path=test_img_path, output_result=output_result, fig_size=(20, 15))
        results.append({'filename': filename, 'bbox_results': bbox_results})
    end_time = time.time()
    duration = end_time - start_time
    eval_results = coco_dataset_evaluation.evaluate(results=results)
    end_timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    logger.info(
        f"eval_results={eval_results}\n{end_timestamp}测试结束,device={device},共在{duration}时间内测试{image_size}张图片：{image_size / duration} pics/sec, {duration * 1000 / image_size} milliseconds")


def get_annotation(image_id,annotations):
    return [annotation  for anno_id,annotation in annotations.items() if annotation['image_id']==image_id]



if __name__ == '__main__':
    # 当启动当前文件为主路口的时候，指定工作目录
    working_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 增加当前路径为 包搜索路径
    sys.path.append(working_dir)
    from object_detection_server.config.configs import *
    from object_detection_server.mmdetection_api import build_logger
    logger = build_logger(log_dir=os.path.dirname(__file__), logger_name= 'object_detection_server')

    # 当从任意位置启动该脚本的时候，需要加入其对应的搜索路径
    from object_detection_server.load_model import model

    data_root = os.path.join(working_dir, 'object_detection_server', 'test_data', 'seal_data_real')
    img_prefix = 'test'

    single_evaluate = False
    if not single_evaluate:
        detection_evaluate(model=model,data_root=data_root)
    else:
        coco_dataset_evaluation = COCODatasetEvaluation(data_root=data_root)
        annotations = coco_dataset_evaluation.coco.anns

        image_idx = 0
        image_id = coco_dataset_evaluation.img_ids[image_idx]
        # for image_idx,image_id in  enumerate(coco_dataset_evaluation.img_ids):
        # 获取每张图片的信息
        img_info = coco_dataset_evaluation.getitem(idx=image_idx)
        filename = img_info['filename']
        image_id_in_annotation = img_info['id']
        # 验证  image_idx  输出的图片 image_id 和标注的数据一致
        assert image_id == image_id_in_annotation
        test_img_path = os.path.join(data_root, img_prefix, filename)
        logger.info(f'image_idx={image_idx},image_id={image_id},test_img_path={test_img_path}')


        # 使用模型进行预测
        test_img_bgr = cv.imread(test_img_path)  # load image as bgr
        bbox_results, processed_image = seal_detection(model=model, image=test_img_bgr)
        annotation = get_annotation(image_id=image_id, annotations=annotations)

        # 对单个数据进行评估
        coco_dataset_evaluation.evaluate_single_result(image_index=image_idx,result=bbox_results)

