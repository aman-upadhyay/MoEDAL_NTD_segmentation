import cv2
import numpy as np

thresh = 0.32
img_binary = np.empty((247, 300, 360, 2))
data = np.load("calibration_data/stacked_clean_foils", allow_pickle=True)
data = np.array(data)
for i in range(0, 247):
	for j in range(0, 2):
		img_binary[i, :, :, j] = cv2.threshold(data[i, :, :, j], thresh, 1, cv2.THRESH_BINARY)[1]
np.save("calibration_data/binary_clean", img_binary)
