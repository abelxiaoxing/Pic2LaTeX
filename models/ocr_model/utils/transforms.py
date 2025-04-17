import numpy as np
import cv2
from typing import List, Union
from PIL import Image
from collections import Counter
from ...globals import (
    IMG_CHANNELS,
    FIXED_IMG_SIZE,
    IMAGE_MEAN, IMAGE_STD,
)

def general_transform_pipeline(image: np.ndarray) -> np.ndarray:

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize image
    target_size = FIXED_IMG_SIZE - 1
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    
    # Add channel dimension for compatibility with ONNX models
    image = np.expand_dims(image, axis=0)
    
    return image


def trim_white_border(image: np.ndarray):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    corners = [tuple(image[0, 0]), tuple(image[0, -1]),
               tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)
    
    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = 15
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(diff) 
    trimmed_image = image[y:y+h, x:x+w]
    return trimmed_image



def padding(images: List[np.ndarray], required_size: int) -> List[np.ndarray]:
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = required_size - h
        pad_w = required_size - w
        padded_img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        padded_images.append(padded_img)
    return padded_images


def inference_transform(images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
    assert IMG_CHANNELS == 1 , "Only support grayscale images for now"
    images = [np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img for img in images]
    images = [trim_white_border(image) for image in images]
    images = [general_transform_pipeline(image) for image in images]
    images = padding(images, FIXED_IMG_SIZE)
    batch = np.stack(images, axis=0)
    
    return batch
