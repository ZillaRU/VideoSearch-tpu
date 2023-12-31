import hashlib
import json

from .utils.video_downloader import generic_web_downloader, get_video_duration
from .clip.scene_features import SceneFeatures

sf = SceneFeatures()

def local_path(path_uri: str) -> str:
    if "http://" not in path_uri and "https://" not in path_uri:
        # we will assume that this is a local path
        return path_uri

    md5 = hashlib.md5(path_uri.encode())
    filename = md5.hexdigest()
    
    return generic_web_downloader(path_uri, '/tmp', filename)

def video_features(model, path_uri: str) -> str:
    system_path = local_path(path_uri)
    return sf.scene_features(model, system_path)

def _video_features(filepaths: list) -> list:
    return [_video_features(f.getvalue()) for f in filepaths]