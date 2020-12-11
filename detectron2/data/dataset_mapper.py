# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from . import detection_utils as utils
from . import transforms as T
from detectron2.config import cfg
import os

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        # image_slice_num: int = 3
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # self.image_slice_num        = image_slice_num
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            # "image_slice_num": cfg.INPUT.SLICE_NUM,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file

        image_fn = dataset_dict["file_name"]
        # bounding boxes
        boxes0 = np.array(dataset_dict['annotations'][0]['bbox'])
        boxes_new_xyxy = np.array([0.0, 0.0, 0.0, 0.0])
        boxes_new_xyxy[0] = boxes0[0].copy()
        boxes_new_xyxy[1] = boxes0[1].copy()
        boxes_new_xyxy[2] = (boxes0[0] + boxes0[2]).copy()
        boxes_new_xyxy[3] = (boxes0[1] + boxes0[3]).copy()
        boxes_new_xyxy -= 1 # coordinates in info file start from 1
        # segmentation
        segmentations0 = np.array(dataset_dict['annotations'][0]['segmentation'])
        segmentations0 -= 1 # coordinates in info file start from 1
        segmentation_new = segmentations0.copy()

        # parameter settings
        spacing3D = np.array(dataset_dict['spacing3D'])
        spacing = spacing3D[0]
        slice_intv = spacing3D[2]  # slice intervals
        diameters = np.array(dataset_dict['diameter'])
        window = np.array(dataset_dict['DICOM_window'])
        gender = float(dataset_dict['gender'] == 'M')
        age = dataset_dict['age']/100
        if np.isnan(age) or age == 0:
            age = .5
        z_coord = dataset_dict['norm_location'][2]
        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE

        if self.is_train and cfg.INPUT.DATA_AUG_3D is not False:
            slice_radius = diameters.min() / 2 * spacing / slice_intv * abs(cfg.INPUT.DATA_AUG_3D)  # lesion should not be too small
            slice_radius = int(slice_radius)
           #  print(slice_radius)
            if slice_radius > 0:
                if cfg.INPUT.DATA_AUG_3D > 0:
                    delta = np.random.randint(0, slice_radius+1)
                  #  print('AUG: ',delta)
                else:  # give central slice higher prob
                    ar = np.arange(slice_radius+1)
                    p = slice_radius-ar.astype(float)
                    delta = np.random.choice(ar, p=p/p.sum())
                   # print('NO AUG: ', delta)
                if np.random.rand(1) > .5:
                    delta = -delta

                img_name = dataset_dict["file_name"].split('/')[-1]
                path_name = dataset_dict["file_name"][:-len(img_name)-1]
                slicename = img_name.split('_')[-1]
                dirname = img_name[:-len(slicename)-1]
                slice_idx = int(slicename[:-4])
                image_fn1 = '%s%s%03d.png' % (dirname, '_', slice_idx + delta)
                if os.path.exists(os.path.join(path_name, image_fn1)):
                    image_fn = os.path.join(path_name, image_fn1)

        # load images
        # image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # image = utils.read_image_cv2(dataset_dict["file_name"], self.image_slice_num)
        image, im_scale, crop = utils.load_pred_img(image_fn, spacing, slice_intv,
                                                    cfg.INPUT.IMG_DO_CLIP, self.is_train, num_slice)

        image -= cfg.INPUT.IMAGE_MEAN

        # clip black border
        if cfg.INPUT.IMG_DO_CLIP:
            # bounding boxes
            offset = [crop[2], crop[0]]
            boxes_new_xyxy -= offset * 2
            # segmentations
            segmentation_new -= offset * 4

        # rescale image
        if im_scale != 1:
            # bounding boxes
            boxes_new_xyxy *= im_scale
            # segmentations
            segmentation_new *= im_scale

        if self.use_instance_mask:
            mask = utils.gen_mask_polygon_from_recist(segmentation_new[0])


        if (not cfg.INPUT.IMG_DO_CLIP) and im_scale == 1:
            utils.check_image_size(dataset_dict, image)

        else:
            boxes_new = [0.0, 0.0, 0.0, 0.0]

            boxes_new[0] = boxes_new_xyxy[0]; boxes_new[1] = boxes_new_xyxy[1]
            boxes_new[2] = boxes_new_xyxy[2] - boxes_new_xyxy[0]
            boxes_new[3] = boxes_new_xyxy[3] - boxes_new_xyxy[1]
            dataset_dict['annotations'][0]['bbox'] = boxes_new

            dataset_dict['annotations'][0]['segmentation'] = segmentation_new.tolist()

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None


        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg


        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            # import gc
            # import objgraph
            #
            # gc.collect()
            # objgraph.show_most_common_types(limit=10)
            # objgraph.show_growth()
            # print('\n')

        return dataset_dict