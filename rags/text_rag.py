import openai
import json
import os
import re
from dotenv import load_dotenv
from pathlib import Path
import traceback
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from recall_utils import load_state
from constants import KNOWLEDGE_BASE_PATH
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.node_parser import SimpleNodeParser
from video_processing.video_rag_qdrant import VideoRagQdrant
from llama_index.embeddings.openai import OpenAIEmbedding
#from langsmith.wrappers import wrap_openai

# Load environment variables
load_dotenv()

endpoint_url = "https://api.openai.com/v1"

configurations = {
    "mistral_7B_instruct": {
        "endpoint_url": os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    "mistral_7B": {
        "endpoint_url": os.getenv("MISTRAL_7B_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-v0.1"
    },
    "openai_gpt-4": {
        "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "audio_model": "tts-1"
    }
}

# Choose configuration
config_key = "openai_gpt-4"
#config_key = "mistral_7B_instruct"
#config_key = "mistral_7B"

# Get selected configuration
config = configurations[config_key]

# Model kwargs
gen_kwargs = {
    "model": config["model"],
    "temperature": 0.2,
    "max_tokens": 500
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "perform_web_search",
            "description": """Get more information about the user's question related to the event. Call this function if you do not have enough information about user's question or the user wants to know more details about the tech event, for example when the user asks 
            'Give more information about the key highlights of the event?' or 
            'Give more details about a particular topic'
            
            Do not output the function arguments twice""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query"
                    },
                    "media_label": {
                        "type": "string",
                        "description": "The media label of the event"
                    }
                },
                "required": ["query", "media_label"],
                "additionalProperties": False,
            }
        }
    }
]

client = openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"])

def load_knowledge_base(media_label):
    # load knowledge base
    knowledge_base = load_state(KNOWLEDGE_BASE_PATH)[media_label]
    input_data = {
        text_path.replace("./", "").replace("//", "/"): media_label
        for text_path in knowledge_base["text_paths"]
    }
    print(f"Knowledge base loaded: {input_data}")
    return input_data

def create_new_index(media_label):
    print(f"Creating or Loading new index for {media_label}")
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    storage_root_path='./events_kb'
    media_storage_path = os.path.join(storage_root_path, media_label_path)
    storage_path = os.path.join(media_storage_path, 'qdrant_mm_db')
    text_storage_path = os.path.join(media_storage_path, 'text_vector_store')
    indices_path = os.path.join(media_storage_path, 'indices')
    data_path = os.path.join(media_storage_path, 'data')
    
    video_rag_inst = VideoRagQdrant(data_path,
        storage_path = storage_path, text_storage_path = text_storage_path,
        text_tsindex_dirpath = indices_path, image_tsindex_dirpath = indices_path)
    video_rag_inst.create_ts_index()
    video_rag_inst.create_vector_index(documents=[])
    video_rag_inst.init_multimodal_oai()
    
    return video_rag_inst

def load_mm_data(video_rag_inst, media_label, media_storage_path, video_paths):
    data_path = os.path.join(media_storage_path, 'data')
    text_shard_path = os.path.join(data_path, 'shards')
    images_path = os.path.join(data_path, 'frames')

    text_docs = []
    all_img_docs = []
    query_str = """Describe each image in context of the video. Give answer in the following JSON format
    {
        "images" : [
            {
                "frame": <file_name from the image document metadata>,
                "description": <frame_description>
            },
                "images" : [
            {
                "frame": <file_name from the image document metadata>,
                "description": <frame_description>
            },
            ...
        ]
    }"""

    context = f"These are the frame of a video from the {media_label} event"
    event_metadata = ""

    for video_filepath in video_paths:
        file_prefix = Path(video_filepath).stem

        video_transcript_shard_path = os.path.join(text_shard_path, file_prefix)
        reader = SimpleDirectoryReader(video_transcript_shard_path, recursive=True)
        print(f"Loading transcript shard from  : {video_transcript_shard_path}")
        for doc in reader.load_data():
            doc.metadata["media_label"] = media_label
            content = json.loads(doc.text)
            doc.text = content['text']
            doc.metadata['timestamps'] = content['timestamps']    
            text_docs.append(doc)

        image_frame_path = os.path.join(images_path, file_prefix)
        img_documents = SimpleDirectoryReader(image_frame_path, recursive=True).load_data()
        print(f"Loading images frames from  : {image_frame_path}")
        for idx in range(0, len(img_documents), 10):
            img_docs_to_procss = img_documents[idx:idx+10]
            print(img_docs_to_procss)
            try:
                text_response = video_rag_inst.query_with_oai(query_str, context, img_docs_to_procss, event_metadata=event_metadata)
            except Exception as e:
                print("Error getting response")
                d = {'images': []}
                for i in range(len(img_docs_to_procss)):
                    d['images'].append({'frame': '', 'description': ''})
                text_response = json.dumps(d)

            print(text_response)
            images_data = json.loads(text_response)
            for i, image_info in enumerate(images_data['images']):
                img_documents[idx+i].text = image_info['description']
                img_path = img_documents[idx+i].metadata['file_path']
                search_path = os.path.join(Path(img_path).parent.name, Path(img_path).name)
                print(search_path)
                img_documents[idx+i].metadata['timestamp'] = video_rag_inst.image_search(search_path)
        all_img_docs.extend(img_documents)
    
    return text_docs, all_img_docs

def save_processed_document(media_label, video_paths, session_state):
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    storage_root_path='./events_kb'
    media_storage_path = os.path.join(storage_root_path, media_label_path)
    storage_path = os.path.join(media_storage_path, 'qdrant_mm_db')
    text_storage_path = os.path.join(media_storage_path, 'text_vector_store')

    
    # Update the index with the new documents
    if media_label in session_state:
        print(f"Index for {media_label} exists in session_state: {session_state}")
        video_rag_inst = session_state[media_label]
    else:
        print(f"Index for {media_label} DOES NOT exist in session_state")
        video_rag_inst = create_new_index(media_label)
        session_state[media_label] = video_rag_inst

    text_docs, img_docs = load_mm_data(video_rag_inst, media_label, media_storage_path, video_paths)
    video_rag_inst.add_documents(video_rag_inst.index, storage_path, text_docs+img_docs)
    video_rag_inst.add_documents(video_rag_inst.text_index, text_storage_path, text_docs)
    print(f"Multimodal Index documents count after adding new documents: {video_rag_inst.count_documents()}")
    print(f"Text Index documents count after adding new documents: {video_rag_inst.count_text_documents()}")
    return text_docs, img_docs

def generate_tags_and_images(media_label, session_state):
    print(f"Generating tags and images for media label: {media_label}")
    storage_root_path='./events_kb'
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    media_storage_path = os.path.join(storage_root_path, media_label_path)
    storage_path = os.path.join(media_storage_path, 'qdrant_mm_db')
    text_storage_path = os.path.join(media_storage_path, 'text_vector_store')
    indices_path = os.path.join(media_storage_path, 'indices')
    data_path = os.path.join(media_storage_path, 'data')

    video_rag_inst = session_state[media_label]

    query_for_metadata = f"What are the key highlights of the {media_label}?"
    img_docs, text_docs = video_rag_inst.retrieve(query_for_metadata)
    
    query_str = f"""Give 3 tags in title case for the {media_label} as a list.
        Suggest 1 image file path, as a title image, from the provided images for the {media_label}
        Give the answer in the following JSON format:
        {{
            "tags": <list of tags>,
            "title_image": <image file path from the provided context>
        }}
    """
    context = f"The following are the key highlights of the {media_label}. Answer user's questions from the provided text and image documents. For the title image, give the exact path stored in img_doc.metadata['file_path'] provided in the img_docs"
    event_metadata = {
        'text_docs': text_docs,
        'img_docs': img_docs
    }
    
    response_text = video_rag_inst.query_with_oai(query_str, context, img_docs, event_metadata=event_metadata)

    try:
        new_metadata = json.loads(response_text)
        
    except Exception as e:
        print(f"Error: Invalid JSON response from OpenAI: {response_text}, {traceback.print_exc()}")
        new_metadata = {
            "tags": [media_label],
            "title_image": None
        }
    else:
        try:
            relative_img_path = new_metadata['title_image'].split('events_kb/', 1)[1]
            new_metadata['title_image'] = os.path.join('events_kb', relative_img_path)
        except Exception as ie:
            print(f"Error processing title image path: {traceback.print_exc()}")
            new_metadata['title_image'] = None 
    print(f"Updating state with new tags and title image: {new_metadata}")
    
    return new_metadata



def load_documents(input_data):
    reader = SimpleDirectoryReader(input_files=input_data.keys())
    documents = []
    for doc in reader.load_data():
        # Assuming doc has a filename or similar attribute
        file_path = doc.metadata.get("file_path", None)  # or however the source is defined
        if file_path is not None:
            doc.metadata["media_label"] = input_data[file_path]
        documents.append(doc)
        
    return documents

async def update_response_container(response_container, response_text, token):
    response_text.append(token)
    response_container.markdown(''.join(response_text))

async def get_llm_response(query, messages, tools_call=True, response_container=None):

    response_text = []
    function_data = {}

    stream = await client.chat.completions.create(
        messages=messages,
        tools=tools if tools_call else None,
        stream=True,
        **gen_kwargs)

    async for part in stream:
        if part.choices[0].delta.tool_calls:
            tool_call = part.choices[0].delta.tool_calls[0]
            function_name = tool_call.function.name or ""
            arguments = tool_call.function.arguments or ""
            index = tool_call.index
            print(f"tool_call: {tool_call}")
            index_data = function_data.setdefault(index, {})
            index_data.setdefault("name", []).append(function_name)
            index_data.setdefault("arguments", []).append(arguments)
        
        if token := part.choices[0].delta.content or "":
            #await update_response_container(response_container, response_text, token)
            response_text.append(token)
            response_container.markdown(''.join(response_text))
    for index, index_data in function_data.items():
        index_data["name"] = ''.join(index_data["name"])
        index_data["arguments"] = ''.join(index_data["arguments"])
    return ''.join(response_text), function_data

def search_knowledge_base(query, media_label, session_state):
    print(f"Query: {query} Media label: {media_label}")
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)

    storage_root_path='./events_kb'
    media_storage_path = os.path.join(storage_root_path, media_label_path)
    storage_path = os.path.join(media_storage_path, 'qdrant_mm_db')
    text_storage_path = os.path.join(media_storage_path, 'text_vector_store')
    indices_path = os.path.join(media_storage_path, 'indices')
    data_path = os.path.join(media_storage_path, 'data')  

    video_rag_inst = session_state[media_label]

    print(f"Index documents count: {video_rag_inst.count_documents()}")
    img_docs, text_docs = video_rag_inst.retrieve(query)

    print(f"Number of relevant text documents: {len(text_docs)}")
    print(f"Number of relevant image documents: {len(img_docs)}")
    print("\n" + "="*50 + "\n")

    return img_docs, text_docs

if __name__ == "__main__":
    media_label = 'Google I/O 2024'
    input_data = load_knowledge_base(media_label)
    documents = load_documents(input_data)
    print(f"Number of documents: {len(documents)}")
    print(f"Query: {query} Media label: {media_label}")
    index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
    print(f"Index count: {len(index.docstore.docs)}")
    retriever = index.as_retriever(retrieval_mode='hybrid', k=10)