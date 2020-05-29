import contextlib
import json
import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
from google.protobuf.json_format import Parse

import datasets.transforms as T
from ouroboros.dgp.datasets import FrameSceneDataset
from ouroboros.dgp.proto import annotations_pb2
from ouroboros.dgp.proto.annotations_pb2 import (BoundingBox2DAnnotation,
                                                 BoundingBox2DAnnotations,
                                                 KeyPoint2DAnnotation,
                                                 KeyPoint2DAnnotations)


class DetectionDataset(FrameSceneDataset):
    def __init__(self,
                 dataset_json_path,
                 split="train",
                 transforms=None,
                 requested_annotations=("bounding_box_2d", )):
        """2D bounding box dataset that inherits from our DGP Detection Dataset

        Parameters
        ----------
        dataset_json_path: str
            Full path to the dataset json holding dataset metadata, ontology, and image and
            annotation paths.
            (cf. `dgp/proto/dataset.proto` for the full details)

        split: str, default: "train"
            Split of dataset to read ("train" | "val" | "test" | "train_overfit")
        """
        super().__init__(scene_dataset_json=dataset_json_path,
                         split=split,
                         datum_names=None,
                         requested_annotations=requested_annotations,
                         requested_autolabels=None,
                         only_annotated_datums=True)
        # TODO: debug build_coco_dataset
        if split in ("train", "val"):
            self.coco = self._build_coco_dataset(requested_annotations[0])

        self.custom_transforms = transforms
        self.id_to_img_map = {i: i for i in range(len(self))}

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        image = sample["rgb"]

        # # Add a dummy annotation if there's no target bbox in the image.
        # if len(sample["bounding_box_2d"]) == 0:
        #     sample["bounding_box_2d"] = [[-1., -1., -1., -1.]]
        #     sample["class_ids"] = [-1]

        # TODO: support other annotation_type
        bbox2d_annotation_list = sample["bounding_box_2d"]
        target = {}
        target["boxes"] = torch.FloatTensor(bbox2d_annotation_list.ltrb)
        target["labels"] = torch.LongTensor(bbox2d_annotation_list.class_ids)
        target["image_id"] = torch.LongTensor([index])
        target["area"] = torch.FloatTensor(
            [box.area for box in bbox2d_annotation_list.boxlist])
        target["iscrowd"] = torch.LongTensor([False] *
                                             len(bbox2d_annotation_list))

        h, w = image.height, image.width
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.custom_transforms is not None:
            image, target = self.custom_transforms(image, target)

        return image, target

    def get_img_info(self, index):
        """Get image metadata by providing the index of the image in the dataset

        Parameters
        ----------
        index: int
            Index of image in dataset \in {0, 1, ..., len(dataset) - 1}

        Returns
        -------
        image_info: dict
            Dictionary containing the following information for image at index `index`
            in this dataset:
                "id": int
                    Image identifier (index of image in `samples` for the requested split)

                "file_name": string
                    Path from `self.dataset_root` to raw image file
                    NOTE: this includes "rgb/" prefix

                "width": int
                    Image width

                "height": int
                    Image height
        """
        return self.image_metadata[index]

    def get_raw_image(self, index):
        sample = super().__getitem__(index)
        image = sample["rgb"]

        return image

    def build_coco_ontology(self, annotation_key):
        sample = super().__getitem__(0)
        ontology = sample[annotation_key].ontology

        # Mapping integer JSON class id's to a contiguous set starting at 1 (as 0 is reserved for background)
        self.json_category_id_to_contiguous_id = OrderedDict(
            (j_id, c_id + 1)
            for c_id, j_id in enumerate(ontology.thing_class_ids))
        # Reverse lookup from contiguous id's to JSON id's
        self.contiguous_category_id_to_json_id = OrderedDict(
            (c_id, j_id)
            for j_id, c_id in self.json_category_id_to_contiguous_id.items())
        id_to_name = ontology._id_to_name
        self.id_to_name = OrderedDict(
            (c_id, id_to_name[j_id])
            for c_id, j_id in self.contiguous_category_id_to_json_id.items())

    def _build_coco_dataset(self, annotation_key="bounding_box_2d"):
        """From dataset item index, build a COCO style validation dataset (for 2D boxes)"""
        self.build_coco_ontology(annotation_key)
        # TODO: debug
        ann_pb2 = annotations_pb2.BOUNDING_BOX_2D if annotation_key == "bounding_box_2d" else annotations_pb2.KEY_POINT_2D
        idx = 0
        self.image_metadata = []
        for (scene_idx, sample_idx_in_scene,
             datum_idx_in_sample) in self.dataset_item_index:
            datum = self.get_datum(scene_idx, sample_idx_in_scene,
                                   datum_idx_in_sample)
            scene_dir = self.get_scene_directory(scene_idx)
            image_datum = datum.datum.image
            self.image_metadata.append({
                "id":
                idx,
                "file_name":
                os.path.basename(image_datum.filename),
                "width":
                image_datum.width,
                "height":
                image_datum.height,
                "annotations":
                os.path.join(scene_dir, image_datum.annotations[ann_pb2])
            })
            idx += 1
        coco = self.get_coco_dataset(self, self.image_metadata,
                                     self.contiguous_category_id_to_json_id,
                                     self.id_to_name, annotation_key)
        return coco

    @staticmethod
    def get_coco_dataset(dataset,
                         image_metadata,
                         contiguous_category_id_to_json_id,
                         id_to_name,
                         annotation_key="bounding_box_2d"):
        """Construct COCO-style annotations JSON with "images", "annotations", and "categories"
        fields and wrap into a pycocotools.coco.COCO dataset.
        The returned object can then be directly used with pycocotools.cocoeval (which is what
        `maskrcnn_benchmark/data/datasets/evaluation/coco/coco_eval` uses)
        Parameters
        ----------
        dataset: torch.utils.Dataset
            Dataset object
        image_metadata:
            Dictionary of item index to image metadata (see `_populate_image_metadata()` function)
        contiguous_category_id_to_json_id: dict
            Reverse lookup from contiguous id's to COCO style JSON id's
        id_to_name: dict
            Map class id's to string names (these id's are contiguous starting at 1)
        Returns
        -------
        coco: pycocotools.coco.COCO
            COCO-style dataset constructed from dataset annotations
        Notes
        -----
        Assumes image_id's to be {0, 1, ..., len(dataset) - 1}
        """
        # TODO: debug
        ann_pb2 = BoundingBox2DAnnotations if annotation_key == "bounding_box_2d" else KeyPoint2DAnnotations
        from pycocotools.coco import COCO
        # Construct "annotations" field
        annotations = []

        # Unique identifier for single bounding box annotation across the entire dataset
        annotation_id = 0
        for image_id in range(len(dataset)):
            annotation_file = os.path.join(
                dataset.dataset_metadata.directory,
                image_metadata[image_id]["annotations"])
            with open(annotation_file, "r") as annotation_file:
                image_annotations = Parse(annotation_file.read(),
                                          ann_pb2()).annotations
                for ann in image_annotations:
                    ## debug
                    if annotation_key == "bounding_box_2d":
                        box = [ann.box.x, ann.box.y, ann.box.w, ann.box.h]
                        iscrowd = ann.iscrowd
                        area = ann.area
                    else:
                        box = [ann.point.x - 3, ann.point.y - 3, 6, 6]
                        iscrowd = False
                        area = 0
                    ## debug
                    annotation_dict = {
                        # 0 indexed, from label
                        "category_id": ann.class_id,
                        "id": annotation_id,
                        "image_id": image_id,
                        "bbox": box,
                        # "iscrowd": ann.iscrowd,
                        "iscrowd": iscrowd,
                        "area": area
                    }
                    annotations.append(annotation_dict)

                    annotation_id += 1

        # Construct "images" field
        images = [image_metadata[i] for i in range(len(dataset))]

        # Construct "categories" field - "supercategory", "isthing", and "color" not required
        categories = [
            {
                # remap ids to zero-index for coco eval (as in label)
                "id": contiguous_category_id_to_json_id[class_id],
                "name": class_name
            } for class_id, class_name in id_to_name.items()
        ]

        coco_annotations = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        # Dump dictionary into a temporary file and then load this file
        # into a pycocotools COCO object
        with tempfile.NamedTemporaryFile() as temp_file:
            filename = temp_file.name
            with open(filename, "w") as _f:
                json.dump(coco_annotations, _f)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                coco = COCO(filename)

        return coco


def make_simple_transforms():
    return T.Compose([
        T.RandomResize([224], max_size=224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build(image_set, args):
    # TODO: support other annotation_type
    # TODO: put customized transformation
    dataset = DetectionDataset(args.dataset_json_path,
                               split=image_set,
                               transforms=make_simple_transforms(),
                               requested_annotations=("bounding_box_2d", ))
    return dataset


if __name__ == "__main__":
    d = DetectionDataset(
        "/mnt/fsx/dgp/tmp_slam_imerit_bulbwise_traffic_light_scene/scene_dataset_v1.5.json"
    )
