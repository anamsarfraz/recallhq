
from llama_index.core.schema import ImageNode
from .video_rag import VideoRag


class VideoRagQdrant(VideoRag):
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
        super().__init__(
            data_path, storage_path = storage_path, text_tsindex_dirpath = text_tsindex_dirpath, image_tsindex_dirpath = image_tsindex_dirpath, use_qdrant=True)
    
    def retrieve_internal(self, retriever_engine, query_str):
        retrieval_results = retriever_engine.retrieve(query_str)

        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node)
            else:
                retrieved_text.append(res_node.node)

        return retrieved_image, retrieved_text 

    def retrieve(self, query_str):
        return self.retrieve_internal(retriever_engine=self.retriever_engine, query_str=query_str)
        
    def query_with_oai(self, query_str, context, img_docs, event_metadata=""):
        text_response = self.openai_mm_llm.complete(prompt=VideoRag._query_prompt.format(
            context_str=context, query_str=query_str, event_metadata=event_metadata), image_documents=img_docs,
            response_format={ "type": "json_object" })

        return text_response.text

