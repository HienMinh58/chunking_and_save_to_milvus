import os
import re 
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text_fixed_overlap(input_file, chunk_size=100, overlap=50, chunk_dir_file_path="chunk_dir_file_path"):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = []
    parts = text.split('--------------------------------------------------')
    words = [word for part in parts for word in part.split()]
    total_words = len(words)
    step = chunk_size - overlap
    for start in range(0, total_words, step):
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        chunk_id = (start // step) + 1
        chunk_file_path = os.path.join(chunk_dir_file_path, f"chunk_{chunk_id}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)
    return chunks

                        
def read_chunk_files(chunk_dir):
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.txt')]
    chunks = []
    for file_name in chunk_files:
        with open(os.path.join(chunk_dir, file_name), 'r', encoding='utf-8') as f:
            content = f.read()
            chunk_id = int(file_name.split('_')[-1].split('.')[0])
            ngay_ban_hanh = re.search(r'ngày \d{1,2} tháng \d{1,2} năm \d{4}', content)
            ngay_ban_hanh = ngay_ban_hanh.group(0) if ngay_ban_hanh else "Không rõ"
            dieu_match = re.search(r'Điều \d+\.', content)
            dieu = dieu_match.group(0) if dieu_match else "Không rõ"
            khoan_match = re.search(r'Khoản \d+\.\d+\.', content)
            khoan = khoan_match.group(0) if khoan_match else "Không rõ"
            
            words = content.split()
            word_count = len(words)
            start_word = 1
            end_word = word_count
            avg_word_per_page = word_count / content.count('--------------------------------------------------') if content.count('--------------------------------------------------') else 1
            start_page = int((chunk_id - 1) * avg_word_per_page) + 1
            end_page = int(chunk_id * avg_word_per_page)
            chunks.append({
                "chunk_id": chunk_id,
                "data": content,
                "metadata": {
                    "ngay_ban_hanh": ngay_ban_hanh,
                    "dieu": dieu,
                    "khoan": khoan,
                    "phan_loai_theo_luat": "Luật Phòng cháy chữa cháy",
                    "word_range": f"Từ {start_word} đến {end_word}",
                    "estimated_page_range": f"Trang {start_page} đến {end_page}"
                }
            })
    return sorted(chunks, key=lambda x: x["chunk_id"])

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk["data"])
        embeddings.append(embedding)
    #print(embeddings)
    return embeddings

def save_to_milvus(chunks, embeddings):
    try:
        connections.connect(host='localhost', port='19530')
        print("Kết nối thành công với localhost")
        if utility.has_collection("chunked_legal_vectors"):
            utility.drop_collection("chunked_legal_vectors")
            print("Đã xóa collection cũ: chunked_legal_vectors")
    except Exception as e:
        print(f"Lỗi localhost: {e}")
    collection_name = "chunked_legal_vectors"
    
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="data_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Chiều vector của all-MiniLM-L6-v2
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="ngay_ban_hanh", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="dieu", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="khoan", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="phan_loai_theo_luat", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="word_range", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="estimated_page_range", dtype=DataType.VARCHAR, max_length=256)
    ]
    schema = CollectionSchema(fields=fields, description="Chunked legal document vectors with metadata")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
    collection.create_index(field_name="embedding", index_params=index_params)
    
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    data_ids = chunk_ids
    datas = [chunk["data"] for chunk in chunks]
    embeddings_list = embeddings
    ngay_ban_hanhs = [chunk["metadata"]["ngay_ban_hanh"] for chunk in chunks]
    dieus = [chunk["metadata"]["dieu"] for chunk in chunks]
    khoans = [chunk["metadata"]["khoan"] for chunk in chunks]
    phan_loai_theo_luats = [chunk["metadata"]["phan_loai_theo_luat"] for chunk in chunks]
    word_ranges = [chunk["metadata"]["word_range"] for chunk in chunks]
    estimated_page_ranges = [chunk["metadata"]["estimated_page_range"] for chunk in chunks]
    
    collection.insert([
        chunk_ids,
        data_ids,
        embeddings_list,
        datas,
        ngay_ban_hanhs,
        dieus,
        khoans,
        phan_loai_theo_luats,
        word_ranges,
        estimated_page_ranges
    ])
    collection.flush()
    collection.load()
    print(f"Đã lưu {len(chunks)} chunks vào Milvus.")
    
base_dir = os.path.dirname(os.path.abspath(__file__))
chunk_dir_file_path = os.path.join(base_dir, "chunk_dir")
input_file = os.path.join(base_dir, "input1.txt")
chunks = chunk_text_fixed_overlap(input_file=input_file, chunk_size=100, overlap=50, chunk_dir_file_path=chunk_dir_file_path)
chunks = read_chunk_files(chunk_dir_file_path)
embeddings = generate_embeddings(chunks=chunks)
save_to_milvus(chunks=chunks, embeddings=embeddings)
collection = Collection("chunked_legal_vectors")
collection.load()
results = collection.query(expr="chunk_id >= 0", output_fields=["chunk_id", "data"])
print(f"Số lượng bản ghi trong collection: {len(results)}")
for result in results:
    print(f"Chunk ID: {result['chunk_id']}, Data: {result['data'][:50]}...")  # In 50 ký tự đầu
