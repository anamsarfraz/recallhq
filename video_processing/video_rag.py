from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index import SimpleDirectoryReader, StorageContext

from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import LanceDBVectorStore


from llama_index import (
    SimpleDirectoryReader,
)

class VideoRag:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def create_index(self):
        self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        self.image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        storage_context = StorageContext.from_defaults(vector_store=self.text_store, image_store=self.image_store)

        # Create the MultiModal index
        self.documents = SimpleDirectoryReader(self.data_path).load_data()

        self.index = MultiModalVectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            )