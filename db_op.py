import json
import leveldb
import uuid
from inference.global_dict import get_value
import pickle
import torch
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
|scene_metadata  |sceneID         |metadata of a scene      |metadata about a scene (start_time, end_time)       |
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

def insert_scene_metadata(sceneID, data):
    b = json.dumps(data)
    level_instance = leveldb.LevelDB('./dbs/scenemetadata_index', create_if_missing=True)
    level_instance.Put(sceneID.encode('utf-8'), b.encode('utf-8'))

def get_scene_metadata_by_id(id):
    level_instance = leveldb.LevelDB('./dbs/scenemetadata_index')
    b = level_instance.Get(bytes(id,'utf-8'))
    res = json.loads(b.decode('utf-8'))
    return res

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
        # level_instance.Put(k.encode('utf-8'), v.encode('utf-8'))
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

"""
Build, update and search faiss index for scene embeddings
"""
def create_faiss_index(scene_list, embedding_list, embeddings_path, index_path):
    results = {'scene_ids': scene_list, 'embedding': embedding_list}
    with open(embeddings_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)    
    # with open(embeddings_path, 'rb') as f:
    #     results = pickle.load(f)
    embeddings = np.array(torch.vstack(embedding_list), dtype=np.float32)
    
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
    
    embeddings = np.array(torch.vstack(results['embedding']), dtype=np.float32)
    index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    # save index
    faiss.write_index(index, index_path)
    # load faiss index
    # index = faiss.read_index(index_path)
    # update faiss index
    # index.add(np.array(torch.vstack(new_embedding_list), dtype=np.float32))
    # index.make_direct_map()
    # index.train(np.array(results['embedding'], dtype=np.float32))
    # index.update()
    # faiss.write_index(index, index_path)

# todo: using faiss
def search_videos(query, model, scene_list, top_n=2, query_mode='text'):
    query_emb = None
    if query_mode == 'text':
        query_emb = model.encode_text(query)[0]
    elif query_mode == 'image':
        query_emb = model.encode_image(query)[0]
    
    if query_emb is not None:
        D, I = get_value('faiss_index').search(query_emb.unsqueeze(0).numpy(), top_n)
        match_list = [scene_list[i] for i in I[0]]

        # calculate number of rows
        # num_rows = -(-top_n // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

        res_path, res_distance = [], []
        
        for sceneID, dis in zip(match_list, D[0]):
            videoID = get_video_by_scene_id(sceneID)
            video_metadata = get_video_metadata_by_id(videoID)
            res_path.append(video_metadata)
            res_distance.append(dis)
        return match_list, res_path, res_distance

    else:
        print('Search Faiss index failed!')
        return None


