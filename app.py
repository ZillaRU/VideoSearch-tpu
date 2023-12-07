from inference.global_dict import _init, get_value, set_value
import os
import streamlit as st
from db_op import *
import sys
from inference.clip_model import load

@st.cache_resource
def load_model(lang):
    model, preprocess = load(name=lang)
    set_value('lang', lang)
    set_value('clip_model', model)
    set_value('clip_img_preprocess', preprocess)

@st.cache_resource
def load_database(lang):
    if os.path.exists(f'./{lang}_faiss_index.index'):
        set_value('faiss_index', faiss.read_index(f'./{lang}_faiss_index.index'))
        print('Faiss index loaded')
    else:
        print('Faiss index not found')
        set_value('faiss_index', None)


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['EN', 'CH']
    
    _init(lang)
    
    load_model(lang)
    load_database(lang)

    from inference.video_features import video_features

    # 创建一个侧边栏
    st.sidebar.title("🤩VideoSearch powered by Airbox")
    # 添加一个选项控件，用于选择当前显示的Tab
    selected_tab = st.sidebar.selectbox("Select", ["Upload Video", "Search Video"])

    VIDEO_COLLECTION = './video_collection'

    # 根据选择的Tab显示不同的内容
    if selected_tab == "Upload Video":
        def add_video_to_index(uploaded_videos):
            if uploaded_videos is not None and uploaded_videos != []:
                for file in uploaded_videos:
                    file_path = os.path.join(VIDEO_COLLECTION, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    st.write(f"File saved to {file_path}")
                    scenes, scene_clip_embeddings = video_features(file_path)
                    
                    # insert video metadata
                    video_id = str(uuid.uuid4())
                    insert_video_metadata(video_id,  file_path)

                    # insert scene2video and video2scene
                    scene_ids = [str(uuid.uuid4()) for _ in range(len(scenes))]
                    insert_scene2video(scene_ids, [video_id * len(scene_clip_embeddings)])
                    insert_video_scene(video_id, scene_ids)

                    # insert scene embeddings
                    create_faiss_index(scene_ids, scene_clip_embeddings, f'./dbs/{lang}/scene_embeddings.pkl', f'./dbs/{lang}/scene_faiss_index.index')
                    load_database(lang)
                    
        with st.form("upload_form"):
            uploaded_videos = st.file_uploader("Select Video", accept_multiple_files=True, type=['mp4', 'avi', 'mov'])
            submit_button = st.form_submit_button("Submit", on_click=add_video_to_index)


                # scene_ids = []
                # for f in scene_clip_embeddings:
                #     scene_id = str(uuid.uuid4())
                #     with open(f, mode='rb') as file:
                #         content = file.read()
                #         insert_scene_embeddings(scene_id, content)
                #     scene_ids.append(scene_id)
                #     scene_embedding_index.insert(scene_id, content)

                # insert_video_scene(video_id, scene_ids)
    elif selected_tab == "视频检索":
        st.title("视频检索")
        set_value('faiss_index', '.')
        # 在form中填入搜索关键字并提交
        with st.form("视频检索"):
            search_query = st.text_input("输入搜索词")
            btn = st.form_submit_button("搜索")
        
            # 处理form提交的函数
            if btn:
                # 在这里添加视频检索的代码
                search_results = search_videos(search_query)
                
                # 显示搜索结果
                for result in search_results:
                    st.write(f"视频名称：{result['name']}")
                    st.write(f"视频路径：{result['path']}")