import os
import re
from pypdf import PdfReader
from chromadb import Collection, chromadb
from sentence_transformers import SentenceTransformer


DOCUMENTS_FOLDER_PATH = "/workspaces/agentic-systems/self_practice/policy_documents/"

def detect_category(file_name):
    if 'refund' in file_name:
        return 'refund'
    elif 'return' in file_name:
        return 'return'
    elif 'insurance' in file_name:
        return 'insurance'
    else:
        return 'general'

def clear_txt(file_txt_content):
    file_content = re.sub(r'\s*\n\s*', ' ', file_txt_content)  # Replace line breaks (and surrounding whitespace) with a single space
    file_content = file_content.strip()
    return file_content

def load_txt_file(full_file_path):

    with open(full_file_path, 'r', encoding='UTF-8') as txt_reader:
        file_content = txt_reader.read()
        cleared_file_content = clear_txt(file_content)
        file_name = os.path.basename(full_file_path)
        file_type = 'txt'
        txt_document = [{
            'file_content': cleared_file_content,
            'metadata': {
                'category': detect_category(file_name),
                'source': file_name,
                'file_type': file_type
            }
        }]
        print(txt_document)
    return txt_document

def load_pdf_file(full_file_path):
    print(f'''LOADING FILE - {full_file_path}''')
    file_name = os.path.basename(full_file_path)
    pdf_reader = PdfReader(full_file_path)
    document = []
    for page_number, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        obj = {
            'file_content': page_text,
            'metadata': {
                'category': detect_category(file_name),
                'source': file_name,
                'file_type': 'pdf',
                'page_number': page_number
            }
        }
        document.append(obj)

    return document



def load_all_documents(folder_path):

    documents = []
    list_of_documents = sorted(os.listdir(folder_path))
    print(list_of_documents)
    documents = []
    for document_name in list_of_documents:
        full_file_path = DOCUMENTS_FOLDER_PATH + document_name
        print(f'''FULL FILE PATH: {full_file_path}''')

        if str(document_name).endswith('.txt'):
            print('Call load_txt_file')
            document = load_txt_file(full_file_path) 
            print(document)
            documents.append(document)
        elif str(document_name).endswith('.pdf'):
            print('Call load_pdf_file')
            document = load_pdf_file(full_file_path)
            print(document)
            documents.append(document)
        else:
            print(f'''Type of file not supported: {document_name}''')
    return documents

# document = [{doc1}, {doc2}]
def chunk_text(doc_id, document):
    print('^^^^^^^^^Chunking Each Document^^^^^^^^^^^')
    chunks = []
    START = 0
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20
    
    for chunk_id, doc in enumerate(document):
       
        END = START + CHUNK_SIZE
        file_content = doc['file_content']
        print(f'''{END} -- {file_content}''')
        if END > len(file_content):
            break
        category = doc['metadata']['category']
        source = doc['metadata']['source']
        chunk_obj = {
            'id': f'''{doc_id}_{chunk_id}_{category}_{source}''',
            'file_content': doc['file_content'][START:END],
            'metadata': doc['metadata']
        }
        chunks.append(chunk_obj)
        START = END - CHUNK_OVERLAP
    print('-=-=-=-=-=-=-=-=-= PRINTING DOC chunk_obj -=-=-=--=-=-=-=-=-=')
    print(chunks)
    return chunks

#  documents = [[{txt_doc1}], [{pdf_doc1_page1}, {pdf_page2}], etc...]
def create_knowledge_base(documents):
    
    chunks_lst = []
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20
    for document_number, document in enumerate(documents):
        chunks = chunk_text(document_number, document)
        print('&&&&&&&&&&&&&&&&CHUNKS CHUNKED')
        print(chunks)
        chunks_lst.append(chunks)

    return chunks_lst

def create_chroma_db():
    client = chromadb.PersistentClient('./chroma_vector_db')
    collection = client.get_or_create_collection('Shopkart_table')
    return collection

def index_chunking(chunks, collection: Collection, model: SentenceTransformer):
    flat_chunks = [chunk for chunk_group in chunks for chunk in chunk_group]

    documents = [item['file_content'] for item in flat_chunks]
    ids = [item['id'] for item in flat_chunks]
    metadatas = [item['metadata'] for item in flat_chunks]
    doc_embeddings = model.encode(documents, convert_to_numpy=True).tolist()

    collection.upsert(
        ids=ids,
        embeddings=doc_embeddings,
        documents=documents,
        metadatas=metadatas
    )


def create_embeddings_model():
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return model

def retrive_chunks_for_user_query(model: SentenceTransformer, collection: Collection, user_query):
    print(f'''User Query: {user_query}''')
    user_embeddings = model.encode(user_query, convert_to_numpy=True).tolist()
    print(f'''User Embeddings: {user_embeddings}''')
    retrieved_chunks = collection.query(query_embeddings=[user_embeddings], n_results=1, include=['metadatas', 'documents', 'distances'])
    return retrieved_chunks

def main():
    
    documents = load_all_documents(DOCUMENTS_FOLDER_PATH)
    print('^^^^^^^^^^^^^^^^^^^^PRINTINGKNOWLEDGE BASE')
    knowledge_base = create_knowledge_base(documents)
    print(knowledge_base)
    collection = create_chroma_db()
    model = create_embeddings_model()
    index_chunking(chunks=knowledge_base, collection=collection, model=model)
    user_query = 'Who is owning the insurance?' 
    retrieved_chunks = retrive_chunks_for_user_query(model, collection, user_query)
    print('*************RETRIEVED CHUNKS*************')
    print(retrieved_chunks)
    # create_grounded_prompt()
    # grounded_response = get_grounded_response_from_llm()
    # print(f'''Formatted and final response from LLM- {grounded_response}''')

if __name__ == "__main__":
    main()