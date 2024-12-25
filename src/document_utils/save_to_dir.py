import os

import PIL.Image

from src.document_utils.doc_parsers import pdf_to_images


def save_image_to_dir(image : PIL.Image.Image, upload_dir:str, name:str = 'im') -> str:

    base_filename = os.path.splitext(name)[0]
    extension = '.png' #os.path.splitext(name)[1]
    filename = name
    counter = 1

    # Проверка существования файла и генерация нового имени, если необходимо
    while os.path.exists(os.path.join(upload_dir, filename)):
        filename = f"{base_filename}_{counter}{extension}"
        counter += 1

    # Сохранение изображения в указанную директорию
    save_path = os.path.join(upload_dir, filename)

    print('---------------------->')
    print(save_path)
    image.save(save_path)

    return save_path


def save_pdf_to_dir_as_images(pdf , upload_dir:str, name:str = 'im') :
    images = pdf_to_images(pdf)

    paths = [save_image_to_dir(image, upload_dir, name) for image in images]

    return images, paths
