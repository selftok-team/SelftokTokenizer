# -*- coding: utf-8 -*-

from ..utils import Registry

BACKBONE_REGISTRY = Registry("backbone")


def build_backbone(cfg, name=None):
    backbone_name = cfg.model.backbone if name is None else name
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)

    return backbone
