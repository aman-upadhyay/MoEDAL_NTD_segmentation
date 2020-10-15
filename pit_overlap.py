import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_laplace
import cv2
import matplotlib.pyplot as plt


def plot(im, x, y):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    ax.plot(y, x, 'go')
    plt.show()


def clean_filter(im, sigma=8):
    z = gaussian_laplace(im, sigma)
    z = z * (z > 0.001)
    z = gaussian_laplace(z, sigma)
    z = (z < -0.00001)
    return z


def pred_filter(im, sigma=8):
    """
    Filter designed to work with images other than unexposed foil images
    """
    z = gaussian_laplace(im, sigma)
    z = z * (z > 0.001)
    z = gaussian_laplace(z, sigma)
    z = (z < -0.00001)
    return z


def xy_fromMask(mask):
    labs, nlab = ndimage.label(mask)
    X, Y = [], []
    for i in range(1, nlab + 1):
        A = (labs == i)
        coords = np.nonzero(A)
        # print(coords)
        x = np.average(coords[0])
        y = np.average(coords[1])
        # print(x,y)
        X.append(x)
        Y.append(y)
    X = [int(x) for x in X]
    Y = [int(y) for y in Y]
    #print(X, Y)
    return X, Y


def test_overlap(X, Y, X2, Y2, tolerance=10):
    """
    Calculate absoloute true positive / negative rates per image pair assuming XY = true
    """
    xnew, ynew = [], []
    for x, y in zip(X, Y):
        for x2, y2 in zip(X2, Y2):
            if (x > (x2 - tolerance)) & (x < (x2 + tolerance)) & (y > (y2 - tolerance)) & (y < (y2 + tolerance)):
                xnew.append(x), ynew.append(y)
    true_pos = len(xnew)
    true_neg = len(X) - len(xnew)  # No. true examples missed
    false_pos = len(X2) - len(xnew)
    return true_pos, true_neg, false_pos


def calc_efficiencies(pred_image_list, true_image_list):
    total_true_pos = 0
    total_true_neg = 0
    total_false_pos = 0

    for pred_image, true_image in zip(pred_image_list, true_image_list):
        true_x, true_y = xy_fromMask(clean_filter(true_image))
        pred_x, pred_y = xy_fromMask(pred_filter(pred_image))

        #plot(true_image, true_x, true_y)  # Optional
        #plot(pred_image, pred_x, pred_y)
        true_pos, true_neg, false_pos = test_overlap(true_x, true_y, pred_x, pred_y, 20)

        total_true_pos += true_pos
        total_true_neg += true_neg
        total_false_pos += false_pos

    total_true = total_true_pos + total_true_neg
    total_pos = total_true_pos + total_false_pos

    print('True positives:', total_true_pos)
    print('False positives:', total_false_pos)
    print('False negatives:', total_true_neg)

    print('Total pos:', total_pos)
    print('Total true:', total_true)


def binary(data, thresh):
    img_binary = np.empty(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[3]):
            img_binary[i, :, :, j] = cv2.threshold(data[i, :, :, j], thresh, 1, cv2.THRESH_BINARY)[1]

    return img_binary


def test():

    true_list = []
    pred_list = []

    clean = np.load("trained_data/backlit_halo_testclean.npy")
    dirty = np.load("trained_data/backlit_halo_predict.npy")
    clean = binary(clean, 0.2)
    dirty = binary(dirty, 0.70)
    for i in range(49):
        true_list.append(clean[i, :, :, 0])
        pred_list.append(dirty[i, :, :, 0])

    calc_efficiencies(pred_list, true_list)


test()
