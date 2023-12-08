# -*- coding: utf-8 -*-
from .clip_model import load
import os

def _init():  # 初始化
    global _global_dict
    _global_dict = {'lang': None}

def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('Fail to fetch '+key)
        return None 
