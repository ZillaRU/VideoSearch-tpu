from inference.global_dict import _init, get_value, set_value
import os
import streamlit as st
from db_op import *
import sys
from inference.clip_model import load, en_tokenize, ch_tokenize
from preload_db import load_faiss_db


# @st.cache_resource
def load_model(lang):
    model, preprocess = load(name=lang)
    set_value('lang', lang)
    set_value('clip_model', model)
    set_value('clip_img_preprocess', preprocess)
    set_value('clip_text_preprocess', en_tokenize if lang == 'EN' else ch_tokenize)
    print('Clip model loaded')

# @st.cache_resource
def load_database(lang):
    if os.path.exists(f'./dbs/{lang}/scene_faiss_index.index') and os.path.exists(f'./dbs/{lang}/scene_embeddings.pkl'):
        index, scene_list = load_faiss_db(f'./dbs/{lang}/scene_faiss_index.index', f'./dbs/{lang}/scene_embeddings.pkl')
        set_value('faiss_index', index)
        set_value('scene_list', scene_list)
        print('Embedding dataset loaded. Faiss index loaded')
    else:
        print('Faiss index not found')
        set_value('faiss_index', None)
        set_value('scene_list', None)


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['EN', 'CH']
    
    _init()
    if get_value('lang') != lang:
        load_model(lang)

    from inference.video_features import video_features

    # 创建一个侧边栏
    st.sidebar.title("🤩VideoSearch powered by Airbox")
    # 添加一个选项控件，用于选择当前显示的Tab
    selected_tab = st.sidebar.selectbox("Select", ["Upload Video", "Search Video by Text", "Search Video by Image"])

    VIDEO_COLLECTION = './video_collection'

    # 根据选择的Tab显示不同的内容
    if selected_tab == "Upload Video":
        st.session_state['add_video'] = None
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
                    insert_video_metadata(video_id, file_path)

                    # insert scene2video and video2scene
                    scene_ids = [str(uuid.uuid4()) for _ in range(len(scenes))]
                    insert_scene2video(scene_ids, [video_id * len(scene_clip_embeddings)])
                    insert_video_scene(video_id, scene_ids)

                    # insert scene embeddings
                    if os.path.exists(f'./dbs/{lang}/scene_embeddings.pkl') and os.path.exists(f'./dbs/{lang}/scene_faiss_index.index'):
                        update_faiss_index(scene_ids, scene_clip_embeddings, f'./dbs/{lang}/scene_embeddings.pkl', f'./dbs/{lang}/scene_faiss_index.index')
                    else:
                        create_faiss_index(scene_ids, scene_clip_embeddings, f'./dbs/{lang}/scene_embeddings.pkl', f'./dbs/{lang}/scene_faiss_index.index')
                    
        with st.form("upload_form"):
            uploaded_videos = st.file_uploader("Select Video", accept_multiple_files=True, type=['mp4', 'avi', 'mov'])
            submit_button = st.form_submit_button("Submit") # , on_click=add_video_to_index)
        if submit_button:
            st.session_state.add_video=True
        
        if st.session_state.add_video:
            add_video_to_index(uploaded_videos)

                # scene_ids = []
                # for f in scene_clip_embeddings:
                #     scene_id = str(uuid.uuid4())
                #     with open(f, mode='rb') as file:
                #         content = file.read()
                #         insert_scene_embeddings(scene_id, content)
                #     scene_ids.append(scene_id)
                #     scene_embedding_index.insert(scene_id, content)

                # insert_video_scene(video_id, scene_ids)
    elif selected_tab == "Search Video by Text":
        # 在form中填入搜索关键字并提交
        if get_value('faiss_index') is None:
            load_database(lang)
            print(get_value('faiss_index'))
        if get_value('faiss_index') is None:
            st.warning('No Faiss index found. Please upload video first.')
        else:
            search_query = st.text_input("Keywords")
            if search_query is not None and not search_query.isspace():
                # 在这里添加视频检索的代码
                search_query = get_value('clip_text_preprocess')(search_query)
                paths, distances = search_videos(search_query, get_value('scene_list'))

                # 显示搜索结果
                for p, d in zip(paths, distances):
                    st.write(f"Video: {p}\nDistance: {d}\n")
                
    elif selected_tab == "Search Video by Image":
        # 在form中填入搜索关键字并提交
        load_database(lang)
        if get_value('faiss_index') is None:
            st.warning('No Faiss index found. Please upload video first.')
        else:
            search_query = None
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, width=256)
                # 在这里添加视频检索的代码
                search_query = get_value('clip_img_preprocess')(img).unsqueeze(0)
                paths, distances = search_videos(search_query, get_value('scene_list'))

                # 显示搜索结果
                for p, d in zip(paths, distances):
                    st.write(f"Video: {p}\nDistance: {d}\n")