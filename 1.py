import cv2
import os

thresh = 75
clean = 'Data_aman/c2_250/'
binary = 'Data_aman/b2_250/'
clean_filename = os.listdir(clean)
for i in clean_filename:
	image = cv2.imread(os.path.join(clean, str(i)))
	img_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
	m = i.replace('c', 'b')
	cv2.imwrite(os.path.join(binary, str(m)), img_binary)
