# -*- coding: utf-8 -*-

import json
import yaml
import logging
import argparse
from easydict import EasyDict


def log_info(s, use_log=False):
    if use_log:
        logging.info(s)
    else:
        print(s)


def write_str_to_txt(file_path, str, mode="a"):
    with open(file_path, mode) as f:
        try:
            f.write(str)
        except:
            f.write("")


def format_file_name(s):
    s = s.replace(" ", "_").replace(".", "").replace(",", "")[:150]
    return s


def write_namespace_to_txt(file_path, json_str, indent=4):
    with open(file_path, "a") as f:
        f.write(json.dumps(vars(json_str), indent=indent))
        f.write("\n")


def read_txt_to_str(file_path):
    with open(file_path, "r") as f:
        info_list = f.read().splitlines()
        return info_list


def read_txt_to_namespace(file_path):
    with open(file_path, "r") as f:
        json_str = json.load(f)
        args = argparse.Namespace(**json_str)
        return args


def replace_txt_str(txt_path, old_str, new_str):
    file_data = ""
    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(txt_path, "w") as f:
        f.write(file_data)


def write_to_yaml(txt_path, value):
    with open(txt_path, "w") as f:
        yaml.dump(value, f)


def read_from_yaml(txt_path):
    with open(txt_path, "r") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return EasyDict(y)
