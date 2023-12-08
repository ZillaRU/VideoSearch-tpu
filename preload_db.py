import faiss
import pickle


def load_faiss_db(faiss_index_path, embeddings_path):
    # load faiss index
    index = faiss.read_index(faiss_index_path)
    print("==================== Faiss is ready. =====================")
    # load embeddings
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)
    scene_list = results['scene_ids']
    print("============== Embedding dataset is ready. ==============")
    return index, scene_list
