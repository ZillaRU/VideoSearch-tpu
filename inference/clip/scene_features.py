import cv2
import typing
import scenedetect as sd
from PIL import Image
import torch
from ..utils.tensor import save_tensor
from ..global_dict import get_value
from transformers import CLIPProcessor, CLIPModel
from ..clip_model.clip import _transform, load
import uuid
from time import strftime, gmtime


class FrameNumTimecode():
    def __init__(self, frame_num: int) -> None:
        self.frame_num = frame_num

class SceneFeatures:
    def __init__(self) -> None:
        self.clip_processor = _transform()

    def collect_scenes_in_video(self, video_path: str) -> typing.List[typing.Tuple[sd.FrameTimecode, sd.FrameTimecode]]:
        video = sd.open_video(video_path)
        sm = sd.SceneManager()
        
        sm.add_detector(sd.ContentDetector(threshold=27.0))
        sm.detect_scenes(video)
        return sm.get_scene_list()

    # def clip_features_to_dic(self, num_of_scenes: int, clip_pixel_scenes: typing.List, scenes: typing.List[typing.Tuple[sd.FrameTimecode, sd.FrameTimecode]]) -> typing.Dict[str, any]:
    #     d = {}
    #     d['num_of_scenes'] = num_of_scenes        
    #     d['clip_image_scenes'] = [{
    #         'local_path': save_tensor(s['image_embeddings']),
    #         'scene': {
    #             'start_frame_num': scenes[s['scene_no']][0].frame_num,
    #             'end_frame_num': scenes[s['scene_no']][1].frame_num,
    #         }
    #     } for s in clip_pixel_scenes]
    #     return d

    def scene_features(self, model, video_path: str, no_of_samples: int = 3) -> typing.Dict:
        print(f'Collecting scenes in {video_path}')
        scenes = self.collect_scenes_in_video(video_path)
        cap = cv2.VideoCapture(video_path)
        scenes_frame_samples = []
        for scene_idx in range(len(scenes)):
            scene_length = abs(scenes[scene_idx][0].frame_num - scenes[scene_idx][1].frame_num)
            every_n = round(scene_length/no_of_samples)
            local_samples = [(every_n * n) + scenes[scene_idx][0].frame_num for n in range(3)]
            
            scenes_frame_samples.append(local_samples)
        
        if len(scenes) == 0:
            # this could denote a single contiguous scene.
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                every_n = round(frame_count/no_of_samples)
                local_samples = [(every_n * n) for n in range(3)]
                scenes_frame_samples.append(local_samples)
                scenes = [(FrameNumTimecode(0), FrameNumTimecode(frame_count))]
        
        FPS = cap.get(5)

        _scenes = []
        for (st, ed) in scenes:
            st = st.frame_num/FPS
            ed = ed.frame_num/FPS
            _scenes.append((st, ed))
        del scenes

        scene_ids = [str(uuid.uuid4()) for _ in range(len(_scenes))]
        scene_clip_embeddings = []
        for scene_idx in range(len(scenes_frame_samples)):
            scene_samples = scenes_frame_samples[scene_idx]

            clip_img_emb_list = []
            for frame_sample in scene_samples:
                cap.set(1, frame_sample)
                ret, frame = cap.read()
                if not ret:
                    print('breaks oops', ret, frame_sample, scene_idx, frame)
                    break
                pil_image = Image.fromarray(frame[:,:,::-1])
                if clip_img_emb_list == []:
                    pil_image.convert("RGB").save(f'./scene_snapshot/{scene_ids[scene_idx]}.jpg')
                clip_pixel_values = self.clip_processor(pil_image)
                clip_img_emb = model.encode_image(clip_pixel_values.unsqueeze(0))
                clip_img_emb_list.append(clip_img_emb)

            scene_clip_embeddings.append(torch.mean(torch.stack(clip_img_emb_list), dim=0))      

        return _scenes, scene_ids, scene_clip_embeddings
