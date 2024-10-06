# Standard library imports
import os

# Third-party imports
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
# Local application imports
from recall_utils import load_state
from constants import KNOWLEDGE_BASE_PATH



# Load environment variables
load_dotenv()

class LocalVS:
    def __init__(self, documents=None, storage_path=os.getenv("LOCALVS_PATH"), embed_model=OpenAIEmbedding(model="text-embedding-ada-002")):
        self.storage_path = storage_path
        if os.path.exists(self.storage_path):
            print(f"Loading index from storage: {self.storage_path}")
            self.storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.index = load_index_from_storage(self.storage_context, embed_model=embed_model)
            if documents is not None:
                self.add_documents(documents)
        else:
            print("Creating new index")
            # Create an empty vector store
            vector_store = SimpleVectorStore()
            # Create a storage context with the empty vector store
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Create an empty VectorStoreIndex
            self.index = VectorStoreIndex.from_documents(documents if documents is not None else [], storage_context=self.storage_context, embed_model=embed_model)
            self.index.storage_context.persist(persist_dir=self.storage_path)
            
    def add_documents(self, documents):
        self.index.refresh_ref_docs(documents)
        self.index.storage_context.persist(persist_dir=self.storage_path)

    def add_document(self, document):
        self.index.insert(document)
        self.index.storage_context.persist(persist_dir=self.storage_path)

    def count_documents(self):
        return len(self.index.docstore.docs)

    def retrieve(self, query, retrieval_mode='similarity', k=5, filters=None):
        retriever = self.index.as_retriever(retrieval_mode=retrieval_mode, k=k, filters=filters)
        #retriever = self.index.as_retriever(retrieval_mode='similarity', k=5)
        return retriever.retrieve(query)