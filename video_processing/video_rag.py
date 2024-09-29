
from llama_index.core.indices import MultiModalVectorStoreIndex

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode

from llama_index.multi_modal_llms.openai import OpenAIMultiModal


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
    def __init__(self, data_path):
        self.data_path = data_path
    
    def create_index(self):
        self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        self.image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        storage_context = StorageContext.from_defaults(vector_store=self.text_store, image_store=self.image_store)

        # Create the MultiModal index
        self.documents = SimpleDirectoryReader(self.data_path, recursive=True).load_data()

        self.index = MultiModalVectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            )
        self.retriever_engine = self.index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)

    def retrieve_internal(self, retriever_engine, query_str):
        retrieval_results = retriever_engine.retrieve(query_str)

        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                display_source_node(res_node, source_length=200)
                retrieved_text.append(res_node.text)

        return retrieved_image, retrieved_text 

    def retrieve_internal_2(self, retriever_engine, query_str):
        return retriever_engine.retrieve(query_str)

    def query_internal(self, query_str):
        return self.retrieve_internal(self.retriever_engine, query_str)

    def retrieve(self, query_str):
        img, txt = self.retrieve_internal(retriever_engine=self.retriever_engine, query_str=query_str)
        image_documents = SimpleDirectoryReader(input_dir=self.data_path, input_files=img).load_data()
        context_str = "".join(txt)
        return context_str, image_documents
    
    def init_multimodal_oai(self):
        self.openai_mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1500)

    def query_with_oai(self, query_str, context, img):
        text_response = self.openai_mm_llm.complete(prompt=VideoRag._query_prompt.format(
        context_str=context, query_str=query_str, event_metadata=""), image_documents=img,)

        print(text_response.text)
        return text_response.text
