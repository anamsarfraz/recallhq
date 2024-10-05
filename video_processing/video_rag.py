import os
import json
from glob import glob
from llama_index.core.indices import MultiModalVectorStoreIndex

from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core.schema import ImageNode

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

import qdrant_client
from tinydb import TinyDB


class VideoRag:
    _query_prompt = (
    "Given the provided information, including relevant images and retrieved context from the video which represents an event, \
 accurately and precisely answer the query without any additional prior knowledge.\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "Additional context for event that the video represents.: {event_metadata} \n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
    )
    def __init__(self, data_path, storage_path = '', text_tsindex_dirpath = None, image_tsindex_dirpath = None, use_qdrant=True):
        self.data_path = data_path
        self.use_qdrant = use_qdrant
        self.text_tsindex_dirpath = text_tsindex_dirpath
        self.image_tsindex_dirpath = image_tsindex_dirpath
        self.storage_path = storage_path
    
    def create_ts_index(self):
        if self.text_tsindex_dirpath:
            text_index_paths = glob(self.text_tsindex_dirpath+'/*_text_tsindex.json')
            self.text_tsindex = TinyDB(os.path.join(self.text_tsindex_dirpath, 'text_tsindex.json'))

            for path in text_index_paths:
                with open(path) as f:
                    self.text_tsindex.insert_multiple(documents=json.load(f)['_default'].values())

        if self.image_tsindex_dirpath:
            img_index_paths = glob(self.image_tsindex_dirpath+'/*_image_tsindex.json')
            self.image_tsindex  = TinyDB(os.path.join(self.image_tsindex_dirpath, 'image_tsindex.json'))
            
            for path in img_index_paths:
                with open(path) as f:
                    self.image_tsindex.insert_multiple(documents=json.load(f)['_default'].values())
    
    def create_vector_index(self, documents=None):
        if self.use_qdrant:
            # Create a local Qdrant vector store
            self.qdrant_client = qdrant_client.QdrantClient(path=self.storage_path)

            self.text_store = QdrantVectorStore(client=self.qdrant_client, collection_name="text_collection")
            self.image_store = QdrantVectorStore(client=self.qdrant_client, collection_name="image_collection")
        else:
            self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
            self.image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        
        doc_store_path = os.path.join(self.storage_path, 'docstore.json')
        if os.path.exists(doc_store_path):
            print(f"Loading index from storage: {self.storage_path}")
            self.storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.index = load_index_from_storage(self.storage_context)
            if documents is not None:
                self.add_documents(documents)
        else:
            # Create an empty vector store
            if not documents and os.path.exists(self.data_path):
                print(f"Creating a new index from the data in {self.data_path}")
                documents = SimpleDirectoryReader(self.data_path, recursive=True).load_data()
            else:
                documents = documents or []
                print(f"Creating a new index from the documents: {len(documents)}")
            storage_context = StorageContext.from_defaults(vector_store=self.text_store, image_store=self.image_store)
            self.index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)
            self.index.storage_context.persist(persist_dir=self.storage_path)

        self.retriever_engine = self.index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)

    def add_documents(self, documents):
        self.index.refresh_ref_docs(documents)
        self.index.storage_context.persist(persist_dir=self.storage_path)

    def add_document(self, document):
        self.index.insert(document)
        self.index.storage_context.persist(persist_dir=self.storage_path)

    def count_documents(self):
        return len(self.index.vector_store.get_nodes())

    def print_text_tsindex(self):
        if self.text_tsindex:
            all_records = self.text_tsindex.all()
            print(f"Number of records = f{len(all_records)}")
            print("-----------")
            print(all_records)
        else:
            print("Text ts index doesn't exist.")

    def print_image_tsindex(self):
        if self.image_tsindex:
            all_records = self.image_tsindex.all()
            print(f"Number of records = f{len(all_records)}")
            print("-----------")
            print(all_records)
        else:
            print("Image ts index doesn't exist.")

    def retrieve_internal(self, retriever_engine, query_str):
        retrieval_results = retriever_engine.retrieve(query_str)

        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                retrieved_text.append(res_node.text)

        return retrieved_image, retrieved_text 

    def retrieve_internal_2(self, retriever_engine, query_str):
        return retriever_engine.retrieve(query_str)

    def query_internal(self, query_str):
        return self.retrieve_internal(self.retriever_engine, query_str)

    def retrieve(self, query_str):
        img, txt = self.retrieve_internal(retriever_engine=self.retriever_engine, query_str=query_str)
        image_documents = SimpleDirectoryReader(input_dir=self.data_path, input_files=img).load_data() if img else []
        context_str = "".join(txt)
        return context_str, image_documents
    
    def init_multimodal_oai(self):
        self.openai_mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1500)

    def query_with_oai(self, query_str, context, img):
        text_response = self.openai_mm_llm.complete(prompt=VideoRag._query_prompt.format(
            context_str=context, query_str=query_str, event_metadata=""), image_documents=img)

        print(text_response.text)
        return text_response.text

