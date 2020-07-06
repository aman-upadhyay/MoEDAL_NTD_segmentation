import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from PIL import Image
from tqdm import tqdm

############################################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [
			tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
	except RuntimeError as e:
		print(e)

############################################################################################
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
	if i < len(clean_dir_filename) * 0.8:
		image_c = Image.open(os.path.join(clean_dir, filename))
		image_b = Image.open(os.path.join(binary_dir, b))
		image_d = Image.open(os.path.join(dirty_dir, d))
		clean_array_80.append(np.asarray(image_c).reshape(300, 360, 1))
		binary_array_80.append(np.asarray(image_b)[:, :, 0].reshape(300, 360, 1))
		dirty_array_80.append(np.asarray(image_d).reshape(300, 360, 1))
	else:
		image_c = Image.open(os.path.join(clean_dir, filename))
		image_b = Image.open(os.path.join(binary_dir, b))
		image_d = Image.open(os.path.join(dirty_dir, d))
		clean_array_20.append(np.asarray(image_c).reshape(300, 360, 1))
		binary_array_20.append(np.asarray(image_b)[:, :, 0].reshape(300, 360, 1))
		dirty_array_20.append(np.asarray(image_d).reshape(300, 360, 1))
# todo add 1 channel if required

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


def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Dirty Image', 'Clean/Binary Image', 'Predicted Image']
	for count in range(len(display_list)):
		plt.subplot(1, len(display_list), count + 1)
		plt.title(title[count])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[count]), cmap="gray")
		plt.axis('off')
	plt.show()


def build_model(input_layer, start_neurons):
	pad = tf.pad(input_layer, [[0, 0], [2, 2], [4, 4], [0, 0]], "CONSTANT")
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(pad)
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)
	pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
	conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)
	pool2 = Dropout(0.5)(pool2)

	conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
	conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
	pool3 = MaxPooling2D((2, 2))(conv3)
	pool3 = Dropout(0.5)(pool3)

	conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
	conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
	pool4 = MaxPooling2D((2, 2))(conv4)
	pool4 = Dropout(0.5)(pool4)

	# Middle
	convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
	convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

	deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(0.5)(uconv4)
	uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
	uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

	deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Dropout(0.5)(uconv3)
	uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
	uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

	deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
	uconv2 = concatenate([deconv2, conv2])
	uconv2 = Dropout(0.5)(uconv2)
	uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
	uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

	deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	uconv1 = Dropout(0.5)(uconv1)
	uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

	output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

	model = tf.keras.Model(input_layer, output_layer)

	return model


input_shape = Input((300, 360, 1))
dijon = build_model(input_shape, 16)
dijon.summary()

dijon.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = pred_mask[..., tf.newaxis]
	return pred_mask[0]


def show_prediction(dataset, top):
	for image, mask in dataset.take(top):
		prediction = dijon.predict(image)
		display([image[0], mask[0, create_mask(prediction)]])


class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_prediction()
		print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


EPOCHS = 10
dijon_history = dijon.fit(train_dataset_c, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset_c,
                          callbacks=[DisplayCallback()])
