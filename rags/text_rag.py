import openai
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
# Load environment variables
load_dotenv()


def load_knowledge_base():
    # load knowledge base
    knowledge_base = load_state(KNOWLEDGE_BASE_PATH)
    input_data = {
        media_paths["text_path"].replace("./", "").replace("//", "/"): media_label 
        for media_label, media_paths in knowledge_base.items()}

    return input_data



def save_processed_document(media_label, input_file):
    reader = SimpleDirectoryReader(input_files=[input_file])
    documents = []
    for doc in reader.load_data():
        doc.metadata["media_label"] = media_label
        documents.append(doc)
    # Update the index with the new documents
    index = LocalVS()
    index.add_documents(documents)
    return documents

def load_documents(input_data):
    input_data = load_knowledge_base()
    reader = SimpleDirectoryReader(input_files=input_data.keys())

    documents = []
    for doc in reader.load_data():
        # Assuming doc has a filename or similar attribute
        file_path = doc.metadata.get("file_path", None)  # or however the source is defined
        if file_path is not None:
            doc.metadata["media_label"] = input_data[file_path]
        documents.append(doc)
        
    return documents

def search_knowledge_base(query, media_label):
    input_data = load_knowledge_base()
    documents = load_documents(input_data)
    print(f"Number of documents: {len(documents)}")

    print(f"Query: {query} Media label: {media_label}")

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="media_label", value=media_label),
        ])

    index = LocalVS()
    print(f"Index documents count: {index.count_documents()}")

    relevant_docs = index.retrieve(query, filters=filters)
    print(f"Number of relevant documents: {len(relevant_docs)}")
    print("\n" + "="*50 + "\n")

    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1}:")
        print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
        print(f"Metadata: {doc.node.metadata}")
        print(f"Score: {doc.score}")
        print("\n" + "="*50 + "\n")

    client = openai.OpenAI()
        
    prompt = f"""
        Given the following context, answer the question:
        {relevant_docs}
        Question: {query}
        """
        
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
    response_text = response.choices[0].message.content
    return response_text

if __name__ == "__main__":
    media_label = "What Is an AI Anyway? | Mustafa Suleyman | TED"    
    query = "What are the dangers of AI?"
    search_knowledge_base(query, media_label)