
import numpy as np
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dot
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D

import keras.backend as KB

from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image


def Encoder(input_shape):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = vgg16.get_layer('block3_pool').output
    x = Conv2D(64, (3, 3), activation='tanh', padding='same', name='encoder_conv1')(x)
    encoded = Conv2D(8, (3, 3), activation='tanh', padding='same', name='encoder_conv2')(x)

    encoder = Model(inputs=vgg16.input, outputs=encoded, name='encoder')
    return encoder


def ConvAE(input_shape):
    img_input = Input(shape=input_shape)

    encoder = Encoder(input_shape)
    encoded = encoder(img_input)

    encoded = Lambda(lambda x: KB.print_tensor(x, 'encoded'))(encoded)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    model = Model(inputs=img_input, outputs=decoded, name='conv_ae')
    return model
