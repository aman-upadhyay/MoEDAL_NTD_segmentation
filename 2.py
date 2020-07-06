import tensorflow as tf
import numpy as np
# from tensorflow import keras
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# from tensorflow_examples.models.pix2pix import pix2pix
import os
from PIL import Image
from tqdm import tqdm

clean_array_80 = list()
binary_array_80 = list()
dirty_array_80 = list()
clean_array_20 = list()
binary_array_20 = list()
dirty_array_20 = list()

clean_dir = "Data_aman/c_250/"
binary_dir = "Data_aman/b_250/"
dirty_dir = "Data_aman/d_250/"

clean_dir_filename = os.listdir(clean_dir)
binary_dir_filename = os.listdir(binary_dir)
dirty_dir_filename = os.listdir(dirty_dir)
i = 0
for filename in tqdm(clean_dir_filename):
	i += 1
	b = filename.replace('c', 'b')
	d = filename.replace('c', 'd')
	if i < len(filename) * 0.8:
		image_c = Image.open(os.path.join(clean_dir, filename))
		image_b = Image.open(os.path.join(binary_dir, b))
		image_d = Image.open(os.path.join(dirty_dir, d))
		clean_array_80.append(np.asarray(image_c))
		binary_array_80.append(np.asarray(image_b)[:, :, 0])
		dirty_array_80.append(np.asarray(image_d))
	else:
		image_c = Image.open(os.path.join(clean_dir, filename))
		image_b = Image.open(os.path.join(binary_dir, b))
		image_d = Image.open(os.path.join(dirty_dir, d))
		clean_array_20.append(np.asarray(image_c))
		binary_array_20.append(np.asarray(image_b)[:, :, 0])
		dirty_array_20.append(np.asarray(image_d))

clean_array_80 = np.array(clean_array_80) / 255
binary_array_80 = np.array(binary_array_80) / 255
dirty_array_80 = np.array(dirty_array_80) / 255
clean_array_20 = np.array(clean_array_20) / 255
binary_array_20 = np.array(binary_array_20) / 255
dirty_array_20 = np.array(dirty_array_20) / 255

TRAIN_LENGTH = dirty_array_80.shape[0]
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_dataset_c = tf.data.Dataset.from_tensor_slices((dirty_array_80, clean_array_80))
test_dataset_c = tf.data.Dataset.from_tensor_slices((dirty_array_20, clean_array_20))
train_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_80, binary_array_80))
test_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_20, binary_array_20))

train_dataset_c = train_dataset_c.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_c = test_dataset_c.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
train_dataset_b = train_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_b = test_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
