import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
import numpy as np
import matplotlib.pyplot as plt

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
		prediction = milos.predict(image)
		for zahlen in range(top):
			display([tf.slice(image[zahlen], [0, 0, 8], [320, 384, 1]), tf.reshape(mask[zahlen], [320, 384, 1]),
			         prediction[zahlen]])


dirty_files = np.load("calibration_data/exposed_foil", allow_pickle=True)
clean_files = np.load("calibration_data/binary_clean.npy", allow_pickle=True)
dirty_files = np.array(dirty_files)
clean_files = np.array(clean_files)

indices = np.random.permutation(dirty_files.shape[0])
training_idx, test_idx = indices[:198], indices[198:]
training_clean, test_clean = clean_files[training_idx, :, :, :], clean_files[test_idx, :, :, :]
training_dirty, test_dirty = dirty_files[training_idx, :, :, :], dirty_files[test_idx, :, :, :]

training_clean = np.pad(training_clean, ((0, 0), (10, 10), (12, 12), (0, 0)), 'constant', constant_values=1)
training_dirty = np.pad(training_dirty, ((0, 0), (10, 10), (12, 12), (0, 0)), 'constant', constant_values=1)
test_clean = np.pad(test_clean, ((0, 0), (10, 10), (12, 12), (0, 0)), 'constant', constant_values=1)
test_dirty = np.pad(test_dirty, ((0, 0), (10, 10), (12, 12), (0, 0)), 'constant', constant_values=1)


def build_model(input_layer, start_neurons):
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
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

	conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
	conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(conv5)
	pool5 = MaxPooling2D((2, 2))(conv5)
	pool5 = Dropout(0.5)(pool5)

	# Middle
	convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool5)
	convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(convm)

	deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv5 = concatenate([deconv5, conv5])
	uconv5 = Dropout(0.5)(uconv5)
	uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)
	uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)

	deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
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


input_shape = Input((320, 384, 10))
milos = build_model(input_shape, 32)
milos.summary()

milos.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

TRAIN_LENGTH = training_dirty.shape[0]
BATCH_SIZE = 16
EPOCHS = 30
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE * EPOCHS)

train_dataset_b = tf.data.Dataset.from_tensor_slices((training_dirty, training_clean[:, :, :, 0]))
test_dataset_b = tf.data.Dataset.from_tensor_slices((test_dirty, test_clean[:, :, :, 0]))
train_dataset_b = train_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_b = test_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)

milos_history = milos.fit(train_dataset_b, epochs=EPOCHS,
                          validation_data=test_dataset_b,
                          callbacks=tf.keras.callbacks.Callback(),
                          verbose=2)

loss = milos_history.history['loss']
val_loss = milos_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('milos:Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 0.03])
plt.legend()
plt.show()

show_prediction(test_dataset_b, 1)
