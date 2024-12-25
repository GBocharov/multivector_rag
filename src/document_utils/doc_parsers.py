from typing import List

import PIL.Image
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes



def pdf_to_images(pdf_path, batch_size : int = 10) -> List[PIL.Image.Image]:
    images = convert_from_path(pdf_path)
    images = [scale_image(im) for im in images]
    #for i, image in enumerate(images):
    #    image.save(f"pages/page_{i + 1}.png", "PNG")
    return images


def bytes_to_images(pdf_bytes, batch_size : int = 10) -> List[PIL.Image.Image]:
    images = convert_from_bytes(pdf_bytes)
    images = [scale_image(im) for im in images]
    #for i, image in enumerate(images):
    #    image.save(f"pages/page_{i + 1}.png", "PNG")
    return images




def data_to_images(file_path, batch_size : int = 10) -> List[PIL.Image.Image]:
    pass




def scale_image(image: Image.Image, new_height: int = 524) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image
