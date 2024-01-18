from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .npuengine import EngineOV
import time


class CLIP():
    def __init__(self,
                 name,
                 is_processing: bool,
                 batch_size: int=1,
                 embed_dim: int=512,
                 # vision
                 image_resolution: int=224,
                 # text
                 transformer_width: int=512,
                 ):
        super().__init__()
        self.name = name
        self.is_processing = is_processing
        if self.name == 'EN':
            if not is_processing:
                self.visual = EngineOV(f'./inference/clip_model/bmodels/{self.name}/clip_vit-b32-imgencoder-1_3_224_224.bmodel')
                # self.visual = EngineOV(f'./inference/clip_model/bmodels/{self.name}/clip_vit-b32-imgencoder-8_3_224_224.bmodel')
                self.text_encoder = EngineOV(f'./inference/clip_model/bmodels/{self.name}/clip-vitb32_textencoder_1684x_fp16_4-77_12ms.bmodel')
                self.text_projection = torch.from_numpy(np.load('./inference/clip_model/bmodels/EN/text_projection512_512.npy'))
                # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            else:
                self.visual = EngineOV(f'./inference/clip_model/bmodels/{self.name}/clip_vit-b32-imgencoder-{batch_size}_3_224_224.bmodel')
        elif self.name == 'CH':
            if not is_processing:
                self.visual = EngineOV(f'./inference/clip_model/bmodels/{self.name}/chinese_clip_imgencoder-1-3-224-224.bmodel')
                self.text_encoder = EngineOV(f'./inference/clip_model/bmodels/{self.name}/chineseclip_vit16_te-1-52_1_52.bmodel')
                # self.logit_scale = 4.6052
            else:
                self.visual = EngineOV(f'./inference/clip_model/bmodels/{self.name}/chinese_clip_imgencoder-{batch_size}-3-224-224.bmodel')
        else:
            raise NotImplementedError
    
    def encode_image(self, image):
        st_time = time.time()
        img_emb = torch.from_numpy(self.visual([image.numpy().astype(np.float32)])[0])
        if not self.is_processing:
            print('====================== Image Encoding: ', time.time() - st_time)
        return img_emb

    def encode_text(self, text):
        assert not self.is_processing
        st_time = time.time()
        if self.name == 'EN':
            x = torch.from_numpy(self.text_encoder([text.numpy().astype(np.int32)])[0])
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        elif self.name == 'CH':
            x = torch.from_numpy(self.text_encoder([text.numpy().astype(np.float32),
                                                     (text != 0).numpy().astype(np.float32)])[0])
        print('====================== Text Encoding: ', time.time() - st_time)
        return x

    # def _encode(self, image, text):
    #     image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text