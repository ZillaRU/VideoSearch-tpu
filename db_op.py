import json
import leveldb
import uuid
from inference.global_dict import get_value
import pickle
import numpy as np
import faiss


"""
Index diagrams

|----------------|----------------|-------------------------|----------------------------------------------------|
|Table           |Key             |Value                    |Description                                         |
|----------------|----------------|-------------------------|----------------------------------------------------|
|video_metadata  |videoID         |metadata of a video      |metadata about a video                              |
|scene_embeddings|sceneID         |CLIP embedding of a scene|tensor data of a specific scene                     |
|video_scene     |videoID         |sceneID list             |referencing which scenes belong to a specific video |
|scene_video     |sceneID         |videoID                  |referencing which video the scene belongs to        |
|----------------|----------------|-------------------------|----------------------------------------------------|

"""
def insert_video_metadata(videoID, data):
    '''
    insert all of the computed metadata from the video into the metadata index, 
    as well as a unique identifier for the video
    '''
    b = json.dumps(data)
    level_instance = leveldb.LevelDB('./dbs/videometadata_index', create_if_missing=True)
    level_instance.Put(videoID.encode('utf-8'), b.encode('utf-8'))

def get_video_metadata_by_id(id):
    level_instance = leveldb.LevelDB('./dbs/videometadata_index')
    b = level_instance.Get(bytes(id,'utf-8'))
    return json.loads(b.decode('utf-8'))

def insert_scene2video(sceneIDs, videoIDs):
    level_instance = leveldb.LevelDB('./dbs/scene_video_index', create_if_missing=True)
    # 创建WriteBatch对象  
    batch = leveldb.WriteBatch()  
    # 添加多个键值对到batch中  
    for k,v in zip(sceneIDs, videoIDs):
        batch.Put(k.encode('utf-8'), v.encode('utf-8'))   
    # 执行批量插入操作  
    level_instance.Write(batch, sync=True) # level_instance.write(batch)

def get_video_by_scene_id(id):
    level_instance = leveldb.LevelDB('./dbs/scene_video_index')
    b = level_instance.Get(bytes(id,'utf-8'))
    return b.decode('utf-8')

def insert_video_scene(videoID, sceneIds):
    b = ",".join(sceneIds)
    level_instance = leveldb.LevelDB('./dbs/video_scene_index', create_if_missing=True)
    level_instance.Put(videoID.encode('utf-8'), b.encode('utf-8'))

def get_scene_by_video_id(id):
    level_instance = leveldb.LevelDB('./dbs/video_scene_index')
    b = level_instance.Get(bytes(id,'utf-8'))
    return b.decode('utf-8').split(',')

def insert_scene_embeddings(sceneID, data):
    level_instance = leveldb.LevelDB('./dbs/scene_embedding_index', create_if_missing=True)
    level_instance.Put(sceneID.encode('utf-8'), data) 

def get_tensor_by_scene_id(id):
    level_instance = leveldb.LevelDB('./dbs/scene_embedding_index')
    b = level_instance.Get(bytes(id,'utf-8'))
    return BytesIO(b)


"""
Build, update and search faiss index for scene embeddings
"""
def create_faiss_index(scene_list, embedding_list, embeddings_path, index_path):
    results = {'scene_ids': scene_list, 'embedding': embedding_list}
    with open(embeddings_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)    
    # with open(embeddings_path, 'rb') as f:
    #     results = pickle.load(f)
    embeddings = np.array(embedding_list, dtype=np.float32)
    
    index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    # save index
    faiss.write_index(index, index_path)

def update_faiss_index(new_scene_list, new_embedding_list, embeddings_path, index_path):
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)
    results['scene_ids'].extend(new_scene_list)
    results['embedding'].extend(new_embedding_list)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    
    # load faiss index
    index = faiss.read_index(index_path)
    # update faiss index
    index.add(np.array(new_embedding_list, dtype=np.float32))
    index.make_direct_map()
    index.train(np.array(results['embedding'], dtype=np.float32))
    index.update()
    faiss.write_index(index, index_path)

# todo: using faiss
def search_videos(query, scene_list, top_n=1, query_mode='text'):
    import pdb; pdb.set_trace()
    query_emb = None
    if query_mode == 'text':
        query_emb = get_value('clip_model').encode_text(query)[0]
    elif query_mode == 'image':
        query_emb = get_value('clip_model').encode_image(query)[0]
    
    if query_emb is not None:
        D, I = get_value('faiss_index').search(query_emb, top_n)
        match_list = [scene_list[i] for i in I[0]]

        # calculate number of rows
        num_rows = -(-num_search // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

        res_path, res_distance = [], []
        
        for i in range(num_rows):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                idx = i*images_per_row + j
                if idx < num_search:
                    sceneID = match_list[idx]
                    videoID = get_video_by_scene_id(sceneID)
                    video_metadata = get_video_metadata(videoID)
                    path = video_metadata['path']
                    distance = D[0][idx]
                    res_path.append(path)
                    res_distance.append(distance)
        return res_path, res_distance

    else:
        return None


