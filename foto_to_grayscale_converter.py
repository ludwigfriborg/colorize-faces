import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

image_size = 64


def convert_image(location):
  start_img = io.imread('lfw_data/images/' + location)
  start_img = resize(start_img, (image_size, image_size))
  final_img = rgb2gray(start_img)

  io.imsave("lfw_converted_smol/images/" + location, final_img)
  io.imsave("lfw_data_smol/images/" + location, start_img)


count = 0
count_max = 9376 # limit max amount images converted

for file in os.listdir("lfw_data/images"):
  if file.endswith(".jpg") and count < count_max:
    convert_image(file)
    count += 1
    print(str(count) + '/' + str(count_max), end='\r')

print('done')
