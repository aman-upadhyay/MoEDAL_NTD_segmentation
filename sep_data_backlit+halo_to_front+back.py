import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
import numpy as np
import matplotlib.pyplot as plt

dirty_files = np.load("calibration_data/exposed_foil", allow_pickle=True)
clean_files = np.load("calibration_data/stacked_clean_foils", allow_pickle=True)
dirty_files = np.array(dirty_files)
clean_files = np.array(clean_files)

indices = np.random.permutation(dirty_files.shape[0])
training_idx, test_idx = indices[:198], indices[198:]
training_clean, test_clean = clean_files[training_idx, :, :, :], clean_files[test_idx, :, :, :]
training_dirty, test_dirty = dirty_files[training_idx, :, :, :], dirty_files[test_idx, :, :, :]
training_dirty = training_dirty[:, :, :, 8:]
test_dirty = test_dirty[:, :, :, 8:]

training_clean = np.pad(training_clean, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
training_dirty = np.pad(training_dirty, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
test_clean = np.pad(test_clean, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)
test_dirty = np.pad(test_dirty, ((0, 0), (2, 2), (4, 4), (0, 0)), 'constant', constant_values=1)


def build_model(input_layer, layer_factor):
	conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(input_layer)
	conv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)
	# pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(pool1)
	conv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)
	# pool2 = Dropout(0.5)(pool2)

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
	# uconv2 = Dropout(0.5)(uconv2)
	uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)
	uconv2 = Conv2D(layer_factor * 2, (3, 3), activation="relu", padding="same")(uconv2)

	deconv1 = Conv2DTranspose(layer_factor * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	# uconv1 = Dropout(0.5)(uconv1)
	uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = Conv2D(layer_factor * 1, (3, 3), activation="relu", padding="same")(uconv1)

	output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(uconv1)

	model = tf.keras.Model(input_layer, output_layer)

	return model


input_shape = Input((304, 368, 2))
backlit_halo = build_model(input_shape, 32)
backlit_halo.summary()

backlit_halo.compile(optimizer='adam',
                     loss="binary_crossentropy",
                     metrics=['accuracy'])

TRAIN_LENGTH = training_dirty.shape[0]
BATCH_SIZE = 16
EPOCHS = 35
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE * EPOCHS)

train_dataset_b = tf.data.Dataset.from_tensor_slices((training_dirty[:, :, :, :], training_clean[:, :, :, :]))
test_dataset_b = tf.data.Dataset.from_tensor_slices((test_dirty[:, :, :, :], test_clean[:, :, :, :]))
train_dataset_b = train_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)
test_dataset_b = test_dataset_b.shuffle(TRAIN_LENGTH).batch(BATCH_SIZE)

backlit_halo_history = backlit_halo.fit(train_dataset_b, epochs=EPOCHS,
                                        validation_data=test_dataset_b,
                                        callbacks=tf.keras.callbacks.Callback(),
                                        verbose=2)

# 5/5 - 8s - loss: 0.0258 - accuracy: 0.9983 - val_loss: 0.0217 - val_accuracy: 0.9986---10 to 1
loss = backlit_halo_history.history['loss']
val_loss = backlit_halo_history.history['val_loss']
accuracy = backlit_halo_history.history['accuracy']
val_accuracy = backlit_halo_history.history['val_accuracy']
epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('backlit_halo:Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0.415, .5])
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('backlit_halo:Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy Value')
plt.ylim([0.5, 0.7])
plt.legend()
plt.show()


def display(display_list):
	plt.figure(figsize=(30, 30))
	title = ['Dirty Image', 'Clean Image Front', 'Clean Image Back', 'Predicted Image Front', 'Predicted Image Back']
	for count in range(len(display_list)):
		plt.subplot(1, len(display_list), count + 1)
		plt.title(title[count])
		plt.imshow(display_list[count], cmap="gray")
		plt.axis('off')
	plt.show()


pr = backlit_halo.predict(test_dirty)

for j in range(49):
	display([test_dirty[j, :, :, 0], test_clean[j, :, :, 0], test_clean[j, :, :, 1], pr[j, :, :, 0], pr[j, :, :, 1]])
