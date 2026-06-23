import os

FILE_PATH = '/workspaces/agentic-systems/self_practice/policy_documents/'
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

def categorize_file_content(file_name: str):
    if 'refunds' in file_name:
        return 'refunds'
    elif 'returns' in file_name:
        return 'returns'
    elif 'shipping' in file_name:
        return 'shipping'
    elif 'warranty' in file_name:
        return 'warranty'
    else:
        return 'general'

def load_txt_file(file_path: str):

    with open(file_path, 'r', encoding='UTF-8') as reader:
        raw_txt = reader.read()
        print('------------ PRINTING RAW TXT ---------------')
        print(raw_txt)
        category = categorize_file_content(file_path)
        print('------- CATEGORIZED AS > ', category)
        return {
            'documet_txt': raw_txt,
            'metadata': {
                'category': category,
                'file_name': os.path.basename(file_path),
                'file_type': 'txt'
            } 
        }

def load_all_policy_docs(file_path: str) -> list[dict:[str, any]]:
    print('$$$$$$$$$$$$$: ', file_path.endswith('.txt'))
    loaded_doc: dict = {}
    if(file_path.endswith('.txt')):
        loaded_doc = load_txt_file(file_path)
        print('========$$$$$$$$$$$========> ', loaded_doc)
    return loaded_doc

def create_knowledge_base():
    
    list_of_file_in_dir = os.listdir(FILE_PATH)
    print(list_of_file_in_dir)
    all_documents: list[dict[str, any]] = []
    for file_name in list_of_file_in_dir:
        final_file_path = FILE_PATH + file_name
        print('=================> ', final_file_path)
        loaded_doc = load_all_policy_docs(final_file_path)
        all_documents.append(loaded_doc)
    print('======> All DOCUMENTS LOADED >>>>>>>', all_documents)

def create_chunks_from_docs(all_docs: list[dict[str, any]]):

    START_INDEX = 0
    END_INDEX = CHUNK_SIZE
    print('INSIDE CREATE CHUNKS FROM DOCS')
    for doc_index, doc in enumerate(all_docs):
        doc_txt = doc.get('document_txt', '')
        print("=======>", doc_txt)
        
        for chunk_index, chunkText in enumera
            text_chunk = doc_txt[START_INDEX: END_INDEX]
            print('%%%%%%%%%%%%%% PRINTING TXT CHUNK:', text_chunk)
            START_INDEX = END_INDEX - CHUNK_OVERLAP
            END_INDEX = START_INDEX + CHUNK_SIZE


    return ""

def main():   
    all_docs = create_knowledge_base()
    all_chunks = create_chunks_from_docs(all_docs)
    
if __name__ == "__main__":
    main()