
import os
import glob
import math
import aaargh
import numpy as np

import h5py
from skimage.util import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, rgba2rgb
from skimage.transform import resize, rotate, rescale

from sklearn.model_selection import train_test_split

from keras.utils import Sequence
from keras.optimizers import Adam
from keras.models import load_model, Model
import keras.backend.tensorflow_backend as KTF
import keras.backend as KB
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras
from tensorflow.python import debug as tf_debug

from model import ConvAE

from keras.applications.imagenet_utils import preprocess_input

from functools import partial


IMAGE_PATH = '../Data/cover_images'
IMG_HEIGHT, IMG_WIDTH = 200, 200
MODEL_WEIGHTS_PATH = './model_files/weights.h5'
MODEL_PATH = './model_files/model.h5'


app = aaargh.App(description='Train model of simliar image detection')


process_input = partial(preprocess_input, data_format='channels_last', mode='tf')


def get_image_file_names():
    return [fname for fname in glob.glob('{}/*/*.jpg'.format(IMAGE_PATH))]


def load_image(fname, img_height, img_width):
    img = imread(fname)
    if len(img.shape) == 2:
        img = gray2rgb(img)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    assert img.shape[2] == 3
    img = resize(img, (img_height, img_width), mode='reflect')
    img = img_as_ubyte(img)
    return img.astype(float)


class ImageSequence(Sequence):

    def __init__(self, image_files, batch_size, img_height, img_width):
        self.image_files = image_files
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return math.ceil(len(self.image_files) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.image_files[idx * self.batch_size: (idx + 1) * self.batch_size]
        images = np.array([load_image(fname, self.img_height, self.img_width) for fname in batch])
        return np.array([process_input(image) for image in images]), np.array([img_as_float(image) for image in images])


def get_train_valid_test_images():
    image_files = get_image_file_names()[:20000]
    train_images, test_images = train_test_split(image_files, test_size=0.1, random_state=42)
    train_images, valid_images = train_test_split(train_images, test_size=0.1, random_state=42)
    return train_images, valid_images, test_images


@app.cmd(name='train', help='Train the model')
@app.cmd_arg('--n-epoch', help='Number of epochs', type=int, default=1)
@app.cmd_arg('--learning-rate', help='Learning rate', type=float, default=0.001)
@app.cmd_arg('--batch-size', help='Size of mini batch', type=int, default=16)
def train(n_epoch, learning_rate, batch_size):
    train_images, valid_images, test_images = get_train_valid_test_images()
    np.random.shuffle(train_images)

    train_sequence = ImageSequence(train_images, batch_size, IMG_HEIGHT, IMG_WIDTH)
    valid_sequence = ImageSequence(valid_images, batch_size, IMG_HEIGHT, IMG_WIDTH)

    model = ConvAE((IMG_HEIGHT, IMG_WIDTH, 3))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_weights(MODEL_WEIGHTS_PATH)
    model.fit_generator(train_sequence, epochs=n_epoch, validation_data=valid_sequence)

    model.save_weights(MODEL_WEIGHTS_PATH)

    KTF.clear_session()


@app.cmd(name='recover', help='Encode-decode given image')
@app.cmd_arg('--image-file', help='Image file name', required=True)
def recover(image_file):
    model = ConvAE((IMG_HEIGHT, IMG_WIDTH, 3))
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)
    else:
        raise ValueError('no model trained')

    image = load_image(image_file, IMG_HEIGHT, IMG_WIDTH)
    image = process_input(image)
    imsave('origin.jpg', np.squeeze(image))
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    print(np.max(output), np.min(output))
    imsave('result.jpg', np.squeeze(output[0]))


@app.cmd(name='similarity', help='Calculate similarities of validation data')
def similarity():
    _, valid_images, _ = get_train_valid_test_images()
    np.random.shuffle(valid_images)

    valid_sequence = ImageSequence(valid_images, 16, IMG_HEIGHT, IMG_WIDTH)

    model = ConvAE((IMG_HEIGHT, IMG_WIDTH, 3))
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)
    else:
        raise ValueError('no model trained')

    model = model.get_layer('encoder')

    embeddings = model.predict_generator(valid_sequence)
    embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    for image, embedding in zip(valid_images, embeddings):
        print(image)
        similarities = np.dot(embeddings, embedding)
        nearest = np.argsort(similarities)[::-1]
        print('nearest:')
        for i in range(1, 6):
            idx = nearest[i]
            print('\t{} sim = {}'.format(valid_images[idx], similarities[idx]))

        print('farest:')
        for i in range(1, 6):
            idx = nearest[-i]
            print('\t{} sim = {}'.format(valid_images[idx], similarities[idx]))

        break


if __name__ == '__main__':
    app.run()
