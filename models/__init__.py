# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args, num_classes=None):
    return build(args, num_classes=num_classes)
