# -*- coding: utf-8 -*-
from .clip_model import load
import os

def _init(lang='EN'):  # 初始化
    global _global_dict
    _global_dict = {}
    model, preprocess = load(name=lang)
    _global_dict['lang'] = lang
    _global_dict['clip_model'] = model
    _global_dict['clip_img_preprocess'] = preprocess
    if os.path.exists(f'./{lang}_faiss_index.index'):
        _global_dict['faiss_index'] = faiss.read_index(f'./{lang}_faiss_index.index')
        print('Faiss index loaded')
    else:
        print('Faiss index not found')
        _global_dict['faiss_index'] = None

def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('Fail to fetch '+key)
        return None 

def reload_faiss_index():
    _global_dict['faiss_index'] = faiss.read_index(f'./{get_value("lang")}_faiss_index.index')
    print('Faiss index reloaded')