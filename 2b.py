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
def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Dirty Image', 'Binary Image', 'Predicted Image']
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
			display([image[zahlen], mask[zahlen], prediction[zahlen]])


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


def build_model(input_layer, layer_factor):
	conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(input_layer)
	conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)
	pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(pool1)
	conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)
	pool2 = Dropout(0.5)(pool2)

	conv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(pool2)
	conv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(conv3)
	pool3 = MaxPooling2D((2, 2))(conv3)
	pool3 = Dropout(0.5)(pool3)

	conv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(pool3)
	conv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(conv4)
	pool4 = MaxPooling2D((2, 2))(conv4)
	pool4 = Dropout(0.5)(pool4)

	# Middle
	convm = Conv2D(layer_factor * 16, (3, 3), activation="relu", padding="same")(pool4)
	convm = Conv2D(layer_factor * 16, (3, 3), activation="relu", padding="same")(convm)

	deconv4 = Conv2DTranspose(layer_factor * 8, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(0.5)(uconv4)
	uconv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(uconv4)
	uconv4 = Conv2D(layer_factor * 8, (3, 3), activation="relu", padding="same")(uconv4)

	deconv3 = Conv2DTranspose(layer_factor * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Dropout(0.5)(uconv3)
	uconv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(uconv3)
	uconv3 = Conv2D(layer_factor * 4, (3, 3), activation="relu", padding="same")(uconv3)

	deconv2 = Conv2DTranspose(layer_factor * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
	uconv2 = concatenate([deconv2, conv2])
	uconv2 = Dropout(0.5)(uconv2)
	uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)
	uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)

	deconv1 = Conv2DTranspose(layer_factor * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	uconv1 = Dropout(0.5)(uconv1)
	uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(uconv1)

	output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

	model = tf.keras.Model(input_layer, output_layer)

	return model


input_shape = Input((304, 368, 1))
dijon = build_model(input_shape, 32)
dijon.summary()

dijon.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

TRAIN_LENGTH = dirty_array_80.shape[0]
BATCH_SIZE = 32
EPOCHS = 20

train_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_80, binary_array_80))
test_dataset_b = tf.data.Dataset.from_tensor_slices((dirty_array_20, binary_array_20))
train_dataset_b = train_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_b = test_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)

dijon_history = dijon.fit(train_dataset_b, epochs=EPOCHS,
                          validation_data=test_dataset_b,
                          callbacks=tf.keras.callbacks.Callback(),
                          verbose=2)

loss = dijon_history.history['loss']
val_loss = dijon_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('dijon:Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 0.1])
plt.legend()
plt.show()

show_prediction(test_dataset_b, 4)
