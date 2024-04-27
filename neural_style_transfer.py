import os

import tensorflow as tf

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image

import tensorflow_hub as hub


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def convert_img(pil_image):
    max_dim = 512

    # Конвертируем PIL изображение в тензор
    img = tf.convert_to_tensor(np.array(pil_image))

    # Убедимся, что изображение имеет 3 канала (RGB)
    if img.shape[-1] != 3:
        img = tf.stack([img, img, img], axis=-1)

    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def convert_img_break(pil_image):
    max_dim = 512
    # Преобразуем PIL изображение в тензор
    img = tf.keras.preprocessing.image.img_to_array(pil_image)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


class StyleStealer:
    def __init__(self, model):
        self.model = model

    def steal_break(self, content_img, style_img):
        return tensor_to_image(self.model(tf.constant(convert_img_break(content_img)),
                                          tf.constant(convert_img_break(style_img)))[0])

    def steal(self, content_img, style_img):
        return tensor_to_image(self.model(tf.constant(convert_img(content_img)),
                                          tf.constant(convert_img(style_img)))[0])


if __name__ == "__main__":
    content_path = "images/input_4.jpg"
    style_path = "images/style_6.png"

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    print("Loading model...")
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    print("Model loaded")
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image).save("output.jpg")
