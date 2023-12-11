# CLIP Video Search
# Please download bmodels and place them in `./inference/clip_model/bmodels`.
```
python -m pip install dfn
# download CLIP VIT-b32 and put these files into ./clip_image_search/clip/bmodels/EN
python3 -m dfn --url https://disk.sophgo.vip/sharing/optDG3uDs
# download ChineseCLIP VIT-16 and put these files into ./clip_image_search/clip/bmodels/CH
python3 -m dfn --url https://disk.sophgo.vip/sharing/qw6hvmVWs
```

[CLIP (Contrastive Language–Image Pre-training)](https://openai.com/blog/clip/) is a technique _which efficiently learns visual concepts from natural language supervision_. CLIP has found applications in [stable diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion).

This repository aims act as a POC in exploring the ability to use CLIP for video search using natural language outlined in the article found [here](https://medium.com/@guyallenross/using-clip-to-build-a-natural-language-video-search-engine-6498c03c40d2).

## Usage
### Dependencies
- python >= 3.8


### Running
`streamlit run app.py EN`

