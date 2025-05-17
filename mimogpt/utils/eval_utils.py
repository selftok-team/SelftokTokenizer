# -*- coding: utf-8 -*-

import torch
import random
import torch.distributed as dist

__all__ = ["RecallAtK_ret", "l2norm", "get_rank", "get_world_size", "caption_shuffle"]


def l2norm(x, dim=-1):
    norm = torch.norm(x, 2, dim, keepdim=True) + 1e-8
    x = torch.div(x, norm)
    return x


def get_rank():
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def caption_shuffle(seg_list, shuffle_percent, start_ratio=0.3, end_ratio=0.7):
    if random.randint(1, 10) <= shuffle_percent * 10:
        # do shuffle caption if satisfied
        seg_len = len(seg_list)
        start = int(seg_len * start_ratio)
        end = int(seg_len * end_ratio)
        if start == end:
            return seg_list
        else:
            # shuffle the caption
            split_index = random.randint(start, end)
            cap1 = seg_list[split_index:]
            cap2 = seg_list[:split_index]
            cap1.extend(cap2)
            return cap1
    else:
        return seg_list


def calc_map(similarity, gt_labels, k=0):
    mAP = 0

    top_idx = torch.argsort(similarity, dim=1, descending=True)
    num_query = similarity.shape[0]
    for i in range(num_query):
        gt_sorted = torch.index_select(gt_labels[i], dim=0, index=top_idx[i])
        gt_sorted = gt_sorted[:k] if k else gt_sorted

        gt_num = torch.sum(gt_sorted != 0)
        if gt_num == 0:
            continue
        gt_count = torch.mul(gt_sorted.cumsum(dim=-1), gt_sorted)
        idx = torch.linspace(1, len(gt_sorted), len(gt_sorted))

        presicion = torch.div(gt_count, idx)
        recall = torch.div(gt_count, gt_num)
        mAP_ = torch.sum(presicion) / gt_num
        mAP = mAP + mAP_

    mAP = mAP / num_query
    return mAP


class BaseMetric:
    def __init__(self, name):
        self.name = name
        # the set of datasets where this metric will be applied
        # an empty set means it will be applied on *all* datasets
        self._dataset_names = set()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, samples, model_output, *args, **kwargs):
        """
        Args:
            samples (Samples): Samples provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def add_applicable_datasets(self, dataset_names):
        self._dataset_names = self._dataset_names.add(dataset_names)

    def is_dataset_applicable(self, dataset_name):
        return len(self._dataset_names) == 0 or dataset_name in self._dataset_names


class RecallAtK_ret(BaseMetric):
    def __init__(self, name, k=1, feat_interaction=0):
        super(RecallAtK_ret, self).__init__(name)
        self.k = k
        self.all_image_features = []
        self.all_text_features = []
        self.feat_interaction = feat_interaction
        if self.feat_interaction == 1 or self.feat_interaction == 2:
            self.all_video_weight_features = []
            self.all_text_weight_features = []
        elif self.feat_interaction == 5:
            self.all_text_hidden_features = []

    def _get_RatK_multi(self, similarity, labels, factor):
        _, top_k_ids = torch.topk(similarity, self.k, dim=1)
        hits = (
            torch.logical_and(labels[:, None] <= top_k_ids, top_k_ids < labels[:, None] + factor).long().max(dim=1)[0]
        )
        return hits

    def calculate(self, similarity, flip=False):
        # calculate image to text retrieval recalls
        # correlations shape is either BxB or Bx(5B)
        # when flip=True, calculate text to image

        if type(similarity) == dict:
            similarity = similarity["similarity"]
        assert similarity.shape[1] % similarity.shape[0] == 0
        batch_size = similarity.shape[0]
        factor = similarity.shape[1] // similarity.shape[0]
        labels = torch.arange(batch_size, device=similarity.device) * factor
        if flip:
            similarity = similarity.t()  # 5B x B
            labels = torch.arange(batch_size, device=similarity.device)
            labels = labels[:, None].expand(-1, factor).flatten()
            factor = 1
        hits = self._get_RatK_multi(similarity, labels, factor)
        ratk = hits.sum().float() / hits.shape[0]
        return ratk

    def collect(self, model_output):
        if self.feat_interaction == 0:
            image_features = model_output["image_features"].detach().cpu()
            text_features = model_output["text_features"].detach().cpu()
            self.all_image_features.append(image_features)
            self.all_text_features.append(text_features)
        else:
            raise NotImplementedError

    def get_total_recall_and_clear(self, model=None):
        all_image_features = torch.cat(self.all_image_features)  # [1000, 1, 512]
        all_text_features = torch.cat(self.all_text_features)  # [1000, 80, 512]

        if self.feat_interaction == 0:
            similarity = all_image_features @ all_text_features.t()
            i2t_recall = self.calculate(similarity)
            t2i_recall = self.calculate(similarity, flip=True)
        else:
            raise NotImplementedError

        self.all_image_features.clear()
        self.all_text_features.clear()

        return i2t_recall, t2i_recall
