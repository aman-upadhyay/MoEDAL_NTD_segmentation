import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Dirty Image', 'Predicted Image', 'Binary Image']
	for count in range(len(display_list)):
		plt.subplot(1, len(display_list), count + 1)
		plt.title(title[count])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[count]), cmap="gray")
		plt.axis('off')
	plt.show()


def show_prediction(dataset, top=1):
	for image, mask in dataset:
		prediction = dijon.predict(image)
		for zahlen in range(top):
			display([image[zahlen], prediction[zahlen], mask[zahlen]])


binary_array_80 = list()
dirty_array_80 = list()
binary_array_20 = list()
dirty_array_20 = list()

binary_dir = "Data_aman/b_250/"
dirty_dir = "Data_aman/d_250/"

binary_dir_filename = os.listdir(binary_dir)
dirty_dir_filename = os.listdir(dirty_dir)
i = 0
for filename in tqdm(binary_dir_filename):
	i += 1
	d = filename.replace('b', 'd')
	if i < len(binary_dir_filename) * 0.8:
		image_b = Image.open(os.path.join(binary_dir, filename))
		image_d = Image.open(os.path.join(dirty_dir, d))
		binary_array_80.append(np.asarray(image_b)[:, :, 0].reshape(300, 360, 1))
		dirty_array_80.append(np.asarray(image_d).reshape(300, 360, 1))
	else:
		image_b = Image.open(os.path.join(binary_dir, filename))
		image_d = Image.open(os.path.join(dirty_dir, d))
		binary_array_20.append(np.asarray(image_b)[:, :, 0].reshape(300, 360, 1))
		dirty_array_20.append(np.asarray(image_d).reshape(300, 360, 1))

binary_array_80 = np.array(binary_array_80) / 255
binary_array_80 = np.pad(binary_array_80, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
dirty_array_80 = np.array(dirty_array_80) / 255
dirty_array_80 = np.pad(dirty_array_80, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
binary_array_20 = np.array(binary_array_20) / 255
binary_array_20 = np.pad(binary_array_20, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
dirty_array_20 = np.array(dirty_array_20) / 255
dirty_array_20 = np.pad(dirty_array_20, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)

TRAIN_LENGTH = dirty_array_80.shape[0]
BATCH_SIZE = 32

train_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_80, binary_array_80))
test_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_20, binary_array_20))
train_dataset_b = train_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_b = test_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)

dijon = tf.keras.models.load_model('saved_model_b/dijon3')

show_prediction(test_dataset_b, 10)
