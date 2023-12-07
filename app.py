from inference.global_dict import _init, get_value, reload_faiss_index
import os
import streamlit as st
from db_op import *
import sys


if __name__ == '__main__':
    lang = sys.argv[1]
    assert lang in ['EN', 'CH']

    if get_value('lang') != lang:
        _init(lang)
    
    from inference.video_features import video_features

    # 创建一个侧边栏
    st.sidebar.title("🤩VideoSearch powered by Airbox")
    # 添加一个选项控件，用于选择当前显示的Tab
    selected_tab = st.sidebar.selectbox("Select", ["Upload Video", "Search Video"])

    VIDEO_COLLECTION = './video_collection'

    # 根据选择的Tab显示不同的内容
    if selected_tab == "Upload Video":
        with st.form("upload_form"):
            uploaded_videos = st.file_uploader("Select Video", accept_multiple_files=True, type=['mp4', 'avi', 'mov'])
            submit_button = st.form_submit_button("Submit")
        
        # 在这里添加处理上传视频的代码
        if submit_button:
            for file in uploaded_videos:
                file_path = os.path.join(VIDEO_COLLECTION, file.name)
                import pdb; pdb.set_trace()
                with open(file_path, "wb") as f:
                    f.write(file.read())
                st.write(f"File saved to {file_path}")
                import pdb; pdb.set_trace()
                scenes, scene_clip_embeddings = video_features(file_path)
                
                # insert video metadata
                video_id = str(uuid.uuid4())
                insert_video_metadata(video_id, {
                    'VideoURI': file_path,
                })

                # insert scene2video and video2scene
                scene_ids = [str(uuid.uuid4()) for _ in range(len(scenes))]
                insert_scene2video(scene_ids, [video_id * len(scene_clip_embeddings)])
                insert_video_scene(video_id, scene_ids)

                # insert scene embeddings
                create_faiss_index(scene_ids, scene_clip_embeddings, f'./dbs/{lang}/scene_embeddings.pkl', f'./dbs/{lang}/scene_faiss_index.index')
                
                reload_faiss_index()
                
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