import keras
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from skimage import color
from skimage import io
from keras import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def create_model():
  model = Sequential()

  model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(64, 64, 3), padding='same')) # set params here

  # Encode
  model.add(Convolution2D(128, (5, 5), activation='relu', padding='same'))
  model.add(MaxPooling2D((2,2)))
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))
  model.add(MaxPooling2D((2,2)))
  

  # Center
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))


  # Decode
  model.add(Convolution2D(256, (5, 5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Convolution2D(128, (5, 5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Convolution2D(64, (5, 5), activation='relu', padding='same'))

  model.add(Convolution2D(3, (1, 1), activation='linear', padding='same'))

  model.compile(loss='mse', optimizer='adam')
  print(model.summary())
  return model


def train(model_name):
  # total img 9376 ~ 32 * 290
  batch_size = 32
  steps_per_epoch = 290
  epochs = 50

  # get model
  # model = create_model()
  model = load_model('m.h5')

  x_datagen = ImageDataGenerator()
  y_datagen = ImageDataGenerator()
  
  seed = 42

  x_generator = x_datagen.flow_from_directory('lfw_converted_smol', batch_size=batch_size, color_mode='rgb', class_mode=None, target_size=(64,64), seed=seed)
  y_generator = y_datagen.flow_from_directory('lfw_data_smol', batch_size=batch_size, color_mode='rgb', class_mode=None, target_size=(64,64), seed=seed)
  train_generator = zip(x_generator, y_generator)

  # fit generator
  model.fit_generator(train_generator, validation_data=train_generator, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch * 0.25, epochs=epochs, verbose=1, max_queue_size=batch_size)

  score = model.evaluate_generator(train_generator, steps=16)
  print(score)

  # save model
  model.save(model_name)


def test(model_name, images=['test.png']):
  # evaluate
  fig = plt.figure(figsize=(7, 10))
  plt.axis('off')

  model = load_model(model_name)
  for i, image in enumerate(images):
    img = color.gray2rgb(io.imread('testfaces/' + image))

    img_res = model.predict(np.expand_dims(img, axis=0))/255

    fig.add_subplot(len(images), 3, 3*i+1, title=image, frameon=False)
    plt.imshow(img)
    fig.add_subplot(len(images), 3, 3*i+2, title='Generated', frameon=False)
    plt.imshow(img_res[0])
    fig.add_subplot(len(images), 3, 3*i+3, title='Combined colorization', frameon=False)
    
    img_res = img_res.reshape((64,64,3))
    img_res = color.rgb2hsv(img_res)
    img_res[:,:,2] = color.rgb2hsv(img)[:,:,2]
    img_res = color.hsv2rgb(img_res)
    plt.imshow(img_res)

  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.show()


# train('m.h5')
test('m.h5', images=['George_W_Bush_0008.jpg', 'Aaron_Sorkin_0001.jpg', 'Helen_Darling_0001.jpg'])
test('m.h5', images=['feynman_s.jpg', 'curie_s.jpg', 'nobel_s.jpg', 'neumann_s.jpg'])
test('m.h5', images=['test_1.png', 'test_2.jpg', 'test_3.jpg', 'test_4.jpg'])