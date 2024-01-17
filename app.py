from inference.global_dict import _init, get_value, set_value
import os
import streamlit as st
from db_op import *
import sys
from inference.clip_model import load, en_tokenize, ch_tokenize
from inference.clip_model.clip import _transform
from preload_db import load_faiss_db
from time import strftime, gmtime
from PIL import Image


@st.cache_resource
def load_model(lang):
    model = load(name=lang)
    print('Clip model loaded')
    return model

# @st.cache_resource
def load_database(lang):
    _init()
    if os.path.exists(f'./dbs/{lang}/scene_faiss_index.index') and os.path.exists(f'./dbs/{lang}/scene_embeddings.pkl'):
        index, scene_list = load_faiss_db(f'./dbs/{lang}/scene_faiss_index.index', f'./dbs/{lang}/scene_embeddings.pkl')
        set_value('faiss_index', index)
        set_value('scene_list', scene_list)
        print('Embedding dataset loaded. Faiss index loaded')
    else:
        print('Faiss index not found')
        set_value('faiss_index', None)
        set_value('scene_list', None)

def query_and_showresults(query, model, scene_list, query_mode, top_n=3):
    query_res = search_videos(search_query, model, get_value('scene_list'), query_mode=query_mode, top_n=top_n)
    if query_res is not None:
        scene_ids, paths, distances = query_res

        for idx in range(len(scene_ids)):
            st.image(f'./scene_snapshot/{scene_ids[idx]}.jpg', width=256)
            st_ed = get_scene_metadata_by_id(scene_ids[idx])
            st.write({
                'video_path': paths[idx],
                'distance': distances[idx],
                'duration': f'{strftime("%H:%M:%S", gmtime(st_ed[0]))} ~ {strftime("%H:%M:%S", gmtime(st_ed[1]))}'
            })


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['EN', 'CH']
    
    model = load_model(lang)

    from inference.video_features import video_features

    # 创建一个侧边栏
    st.sidebar.title("🤩VideoSearch powered by Airbox")
    # 添加一个选项控件，用于选择当前显示的Tab
    selected_tab = st.sidebar.selectbox("Select", ["Upload Video", "Search Video by Text", "Search Video by Image"])

    VIDEO_COLLECTION = './video_collection'

    # 根据选择的Tab显示不同的内容
    if selected_tab == "Upload Video":
        
        clear_button = st.sidebar.button("Clean all video collections")
        if clear_button:
            os.system(f'rm -rf {VIDEO_COLLECTION}/*')
            os.system(f'rm -rf ./scene_snapshot/*')
            os.system(f'rm -rf ./dbs/{lang}/*')
            os.system(f'rm -rf ./dbs/*_index')
            st.write('All video collections are removed.')
        
        st.session_state['add_video'] = None
        def add_video_to_index(uploaded_videos):
            if uploaded_videos is not None and uploaded_videos != []:
                for file in uploaded_videos:
                    file_path = os.path.join(VIDEO_COLLECTION, file.name)
                    if os.path.exists(file_path):
                        st.warning(f"File {file_path} already exists. Skipping...")
                        return
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    st.write(f"File saved to {file_path}")
                    scenes, scene_ids, scene_clip_embeddings = video_features(model, file_path)
                    print(f'{len(scenes)} scenes are detected in {file_path}')
                    
                    # insert video metadata
                    video_id = str(uuid.uuid4())
                    insert_video_metadata(video_id, file_path)

                    # insert scene2video and video2scene
                    insert_scene2video(scene_ids, [video_id] * len(scene_clip_embeddings))
                    insert_video_scene(video_id, scene_ids)

                    # insert scene metadata
                    for scene_id, scene in zip(scene_ids, scenes):
                        insert_scene_metadata(scene_id, scene)

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

    elif selected_tab == "Search Video by Text":
        # 在form中填入搜索关键字并提交
        if get_value('faiss_index') is None:
            load_database(lang)
            print(get_value('faiss_index'))
        if get_value('faiss_index') is None:
            st.warning('No Faiss index found. Please upload video first.')
        else:
            search_query = st.text_input("Keywords")
            if search_query is not None and search_query!= '' and not search_query.isspace():
                # 在这里添加视频检索的代码
                search_query = en_tokenize(search_query) if lang == 'EN' else ch_tokenize(search_query)
                query_and_showresults(search_query, model, get_value('scene_list'), query_mode='text')
    elif selected_tab == "Search Video by Image":
        # 在form中填入搜索关键字并提交
        load_database(lang)
        if get_value('faiss_index') is None:
            st.warning('No Faiss index found. Please upload video first.')
        else:
            search_query = None
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, width=256)
                # 在这里添加视频检索的代码
                preprocess = _transform()
                search_query = preprocess(img).unsqueeze(0)
                query_and_showresults(search_query, model, get_value('scene_list'), query_mode='image')
