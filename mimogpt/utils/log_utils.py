# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import numpy as np
import torch.distributed as dist
from datetime import datetime

__all__ = ["hf_logger", "Registry", "Timer", "time_since"]


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def _register(self, obj, name=None):
        if name is None:
            name = obj.__name__

        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        if obj is not None:
            self._register(obj=obj, name=name)
            return obj

        def _register_cls(cls):
            self._register(obj=cls, name=name)
            return cls

        return _register_cls

    def get(self, name: str):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

# class AverageMeterList(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, length=0, listlen=1, fstr=""):
#         self.length = length
#         self.fstr = fstr

#         if self.length > 0:
#             self.history = []
#         else:
#             self.count = 0
#             self.sum = 0.0
#         self.val = [0.0] * listlen
#         self.avg = 0.0 * listlen
#         self.reset()

#     def reset(self):
#         if self.length > 0:
#             self.history = []
#         else:
#             self.count = 0
#             self.sum = 0.0
#         self.val = [0.0] * listlen
#         self.avg = [0.0] * listlen

#     def reduce_update(self, tensor, num=1):
#         dist.all_reduce(tensor)
#         self.update(tensor.item(), num=num)

#     def reduce_update_group(self, tensor, num=1, group=None):
#         if not group:
#             dist.all_reduce(tensor)
#         else:
#             dist.all_reduce(tensor, group=group)
#         self.update(tensor.item(), num=num)

#     def update(self, val, num=1):
#         if self.length > 0:
#             # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
#             assert num == 1
#             self.history.append(val)
#             if len(self.history) > self.length:
#                 del self.history[0]

#             self.val = self.history[-1]
#             self.avg = np.mean(self.history)
#         else:
#             self.val = val
#             self.sum += val * num
#             self.count += num
#             self.avg = self.sum / self.count

#     def get_val_str(self):
#         if len(self.fstr) > 0 and self.fstr.startswith("%"):
#             return self.fstr % self.val
#         else:
#             return str(self.val)

#     def get_avg_str(self):
#         if len(self.fstr) > 0 and self.fstr.startswith("%"):
#             return self.fstr % self.avg
#         else:
#             return str(self.avg)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0, fstr=""):
        self.length = length
        self.fstr = fstr

        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update_list(self, tensor, num=1):
        dist.all_reduce(tensor)
        self.update_list(tensor.cpu().tolist(), num=num)


    def reduce_update(self, tensor, num=1):
        dist.all_reduce(tensor)
        self.update(tensor.item(), num=num)

    def reduce_update_group(self, tensor, num=1, group=None):
        if not group:
            dist.all_reduce(tensor)
        else:
            dist.all_reduce(tensor, group=group)
        self.update(tensor.item(), num=num)

    def update_list(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history, 0)
        else:
            raise NotImplementedError


    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    def get_val_str(self):
        if len(self.fstr) > 0 and self.fstr.startswith("%"):
            return self.fstr % self.val
        else:
            return str(self.val)

    def get_avg_str(self):
        if len(self.fstr) > 0 and self.fstr.startswith("%"):
            return self.fstr % self.avg
        else:
            return str(self.avg)


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception("{} is not in the clock.".format(key))
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


def time_since(last_time):
    time_elapsed = time.time() - last_time
    current_time = time.time()
    return current_time, time_elapsed


class HF_LOGGER(logging.Logger):
    def __init__(self, logger_name):
        super(HF_LOGGER, self).__init__(logger_name)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s")
        console.setFormatter(formatter)
        self.addHandler(console)
        self.rank = 0

    def setup_logging_file(self, log_dir, rank=0):
        self.rank = rank
        if self.rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_name = datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S") + ".log"
            log_fn = os.path.join(log_dir, log_name)
            fh = logging.FileHandler(log_fn)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            fh.setFormatter(formatter)
            self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info("Args:")
        if isinstance(args, (list, tuple)):
            for value in args:
                self.info("--> {}".format(value))
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                self.info("--> {}: {}".format(key, args_dict[key]))
        self.info("")


logger_name = "mimogpt"
hf_logger = HF_LOGGER(logger_name)
