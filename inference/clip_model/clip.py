import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import CLIP

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
_en_tokenizer = _Tokenizer()
def en_tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _en_tokenizer.encoder["<|startoftext|>"]
    eot_token = _en_tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _en_tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


from tokenizers import Tokenizer
_tokenizer = Tokenizer.from_file("./inference/clip_model/saved_tokenizer/bert_chinese_tokenizer-fast/fast_tokenizer.json")
def cn_tokenize(texts: Union[str, List[str]], context_length: int = 52, truncate: bool = True) -> Union[torch.IntTensor, torch.LongTensor]:
    tokens_and_encodings = _tokenizer.encode_batch(
        texts,
        add_special_tokens=True,
        is_pretokenized=False,
    )
    input_ids = tokens_and_encodings[0].ids
    if len(input_ids) > context_length:
        if truncate:
            input_ids = input_ids[:context_length]
        else:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
    else:
        input_ids += [0] * (context_length - len(input_ids))
    return torch.tensor(input_ids).unsqueeze(0)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: str="tpu", batch_size: int=1, processing=False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    model = CLIP(name=name, batch_size=batch_size, is_processing=processing)
    return model, _transform(n_px=224)