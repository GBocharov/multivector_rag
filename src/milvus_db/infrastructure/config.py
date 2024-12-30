import os

current_dir = os.getcwd()

milvus_db_save_dir = os.path.join(current_dir,'..' ,'milvus_data')

milvus_image_data_save_dir = os.path.join(current_dir, '..' , 'milvus_data/images')

milvus_document_data_save_dir = os.path.join(current_dir, '..' , 'milvus_data/documents')


os.makedirs(name=milvus_db_save_dir, exist_ok = True)

os.makedirs(name=milvus_image_data_save_dir, exist_ok = True)

os.makedirs(name=milvus_document_data_save_dir, exist_ok = True)

