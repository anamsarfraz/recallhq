import openai
import json
import os
import re
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from utils import load_state
from constants import KNOWLEDGE_BASE_PATH
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.node_parser import SimpleNodeParser
from vector_stores.local_vs import LocalVS
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

def save_processed_document(media_label, input_files):
    print(f"Saving processed document in index: {input_files}")
    reader = SimpleDirectoryReader(input_files=input_files)
    documents = []
    for doc in reader.load_data():
        doc.metadata["media_label"] = media_label
        documents.append(doc)
    # Update the index with the new documents
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    index = LocalVS(storage_path=os.path.join(os.getenv("LOCALVS_PATH"), media_label_path))
    index.add_documents(documents)
    print(f"Index documents count after adding new documents: {index.count_documents()}")
    return documents

def generate_tags(media_label):
    print(f"Generating tags for media label: {media_label}")
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    index = LocalVS(storage_path=os.path.join(os.getenv("LOCALVS_PATH"), media_label_path))
    query_for_tags = f"What are the key highlights of the {media_label}?"
    relevant_docs = index.retrieve(query_for_tags, retrieval_mode='hybrid', k=10)
    
    prompt_for_tags = f"The following are the key highlights of the {media_label}: {relevant_docs}. \
        Give 3 tags for the {media_label} as a list. \
        Give the answer in JSON format"

    client = openai.OpenAI(api_key=config["api_key"], base_url=config["endpoint_url"])
    response = client.chat.completions.create(
        model=config["model"],    
        messages=[{"role": "user", "content": prompt_for_tags}],
        temperature=0.2,
        response_format={ "type": "json_object" }
    )
    response_text = response.choices[0].message.content
    try:
        new_tags = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON response from OpenAI: {response_text}")
        new_tags = [media_label]
    print(f"Updating state with new tags: {new_tags}")
    
    return new_tags



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

def search_knowledge_base(query, media_label):
    print(f"Query: {query} Media label: {media_label}")
    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    
    index = LocalVS(storage_path=os.path.join(os.getenv("LOCALVS_PATH"), media_label_path), embed_model=embedding_model)
    print(f"Index documents count: {index.count_documents()}")
    relevant_docs = index.retrieve(query, retrieval_mode='hybrid', k=10)

    print(f"Number of relevant documents: {len(relevant_docs)}")
    print("\n" + "="*50 + "\n")

    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1}:")
        print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
        print(f"Metadata: {doc.node.metadata}")
        print(f"Score: {doc.score}")
        print("\n" + "="*50 + "\n")

    return relevant_docs

if __name__ == "__main__":
    media_label = 'Google I/O 2024'
    input_data = load_knowledge_base(media_label)
    documents = load_documents(input_data)
    print(f"Number of documents: {len(documents)}")
    print(f"Query: {query} Media label: {media_label}")
    index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
    print(f"Index count: {len(index.docstore.docs)}")
    retriever = index.as_retriever(retrieval_mode='hybrid', k=10)