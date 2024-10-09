
from video_processing.video_rag import VideoRag

from llama_index.core.schema import ImageNode, ImageDocument


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
    def __init__(self, data_path, storage_path = '', text_storage_path = '', text_tsindex_dirpath = None, image_tsindex_dirpath = None, use_qdrant=True):
        super().__init__(
            data_path, storage_path = storage_path, text_storage_path = text_storage_path, text_tsindex_dirpath = text_tsindex_dirpath, image_tsindex_dirpath = image_tsindex_dirpath, use_qdrant=True)
    
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
        img_docs = []
        text_docs = []
        img_nodes, text_nodes = self.retrieve_internal(retriever_engine=self.retriever_engine, query_str=query_str)


        for node in img_nodes:
            img_docs.append(ImageDocument(text=node.text, image_mimetype=node.image_mimetype, metadata=node.metadata))

        for node in text_nodes:
            text_docs.append(
                {
                    'text': node.text,
                    'file_name': node.metadata['file_name'],
                    'file_path': node.metadata['file_path'],
                    'timestamps': node.metadata['timestamps']
                }
            )

        text_only_results = self.text_retriever_engine.retrieve(query_str)
        print(f"Text only retrieval results: {text_only_results}")

        for node in text_only_results:
            print(node)
            print(node.metadata)
            text_docs.append(
                {
                    'text': node.text,
                    'file_name': node.metadata['file_name'],
                    'file_path': node.metadata['file_path'],
                    'timestamps': node.metadata['timestamps']
                }
            )

        return img_docs, text_docs
        
    def query_with_oai(self, query_str, context, img_docs, event_metadata=""):
        text_response = self.openai_mm_llm.complete(prompt=VideoRag._query_prompt.format(
            context_str=context, query_str=query_str, event_metadata=event_metadata), image_documents=img_docs,
            response_format={ "type": "json_object" })

        return text_response.text

