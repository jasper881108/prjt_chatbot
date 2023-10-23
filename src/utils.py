import os
from functools import partial
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy

def read_or_create_knowledge_database(knowledge_path, db_path, embeddings, text_splitter):
    if os.path.exists(db_path):
        vectordb = FAISS.load_local(db_path, embeddings)
        return vectordb
    else:
        if not isinstance(knowledge_path, list):
            knowledge_path = [knowledge_path]

        all_docs = []
        for path in knowledge_path:
            documents = TextLoader(path).load()
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)

        vectordb = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE) 
        vectordb.save_local(db_path)

    return vectordb

def parse_config_and_return_vectordb_dict(knowledge_path_list, db_path_list, db_name_list, embeddings, text_splitter):

    vectordb_dict = dict()
    for knowledge_path, db_path, db_name in list(zip(knowledge_path_list, db_path_list, db_name_list)):
        vectordb = read_or_create_knowledge_database(knowledge_path, db_path, embeddings, text_splitter)
        vectordb_dict[db_name] = vectordb
    
    return vectordb_dict