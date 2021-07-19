
import numpy as np
from matplotlib import pyplot as plt
import global_variables
from keras import backend as K
import tensorflow as tf
from DEXTR.helpers import helpers as helpers
from DEXTR.networks.dextr import DEXTR
import cv2

def crop_images(image_1, image_2):
    scale = 0.2
    image_1_small = cv2.resize(image_1, (0,0), fx=scale, fy=scale , interpolation=cv2.INTER_AREA)
    image_2_small = cv2.resize(image_2, (0,0), fx=scale, fy=scale , interpolation=cv2.INTER_AREA)
    modelName = 'dextr_pascal-sbd'
    pad = 50
    thres = 0.8

    # Handle input and output args
    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=modelName,
                    num_input_channels=4, classifier='psp', sigmoid=True)

        plt.figure()
        plt.ion()
        plt.axis('off')
        plt.imshow(image_1_small, cmap='gray')
        plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')
        plt.show()
        #################----image1----##############################################################################
        results_1 = []
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image_1_small, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image_1_small, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [
            pad,
            pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result_1 = helpers.crop2fullmask(pred, bbox, im_size=image_1_small.shape[:2], zero_pad=True, relax=pad) > thres

        results_1.append(result_1)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image_1_small / 255, results_1))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
        result_1 = np.asarray(result_1, dtype="uint8")
        result_1 = cv2.resize(result_1,(image_1.shape[1],image_1.shape[0]) ,interpolation=cv2.INTER_AREA)
        for i in range(image_1.shape[:2][0]):
            for j in range(image_1.shape[:2][1]):
                if result_1[i][j] == False:
                    image_1[i][j] = 0
        cv2.imwrite(global_variables.output_dir + '/cropped_1.jpg', image_1)
        #################----image2----##############################################################################

        plt.figure()
        plt.ion()
        plt.axis('off')
        plt.imshow(image_2_small, cmap='gray')
        plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')
        plt.show()

        results_2 = []
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image_2_small, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image_2_small, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [
            pad,
            pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result_2 = helpers.crop2fullmask(pred, bbox, im_size=image_2_small.shape[:2], zero_pad=True, relax=pad) > thres
        results_2.append(result_2)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image_2_small / 255, results_2))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
        result_2 = np.asarray(result_2, dtype="uint8")
        result_2 = cv2.resize(result_2, (image_2.shape[1],image_2.shape[0]), interpolation=cv2.INTER_AREA)
        for i in range(image_2.shape[:2][0]):
            for j in range(image_2.shape[:2][1]):
                if result_2[i][j] == False:
                    image_2[i][j] = 0
        if (global_variables.save_extra_stuff):
            cv2.imwrite(global_variables.output_dir + '/cropped_2.jpg', image_2)
        return image_1, image_2, result_1, result_2


