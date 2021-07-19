#!/usr/bin/env python3

"""
@license: Apache License Version 2.0
@author: Stefano Di Martino
Exact histogram matching
"""


import numpy as np
from scipy import signal


class ExactHistogramMatcher:
    _kernel1 = 1.0 / 5.0 * np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]])

    _kernel2 = 1.0 / 9.0 * np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]])

    _kernel3 = 1.0 / 13.0 * np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 1, 0, 0]])

    _kernel4 = 1.0 / 21.0 * np.array([[0, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 0]])

    _kernel5 = 1.0 / 25.0 * np.array([[1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]])
    _kernel_mapping = {1: [_kernel1],
                       2: [_kernel1, _kernel2],
                       3: [_kernel1, _kernel2, _kernel3],
                       4: [_kernel1, _kernel2, _kernel3, _kernel4],
                       5: [_kernel1, _kernel2, _kernel3, _kernel4, _kernel5]}

    @staticmethod
    def get_histogram(image, image_bit_depth=8):
        """
        :param image: image as numpy array
        :param image_bit_depth: bit depth of the image. Most images have 8 bit.
        :return:
        """
        max_grey_value = pow(2, image_bit_depth)

        if len(image.shape) == 3:
            dimensions = image.shape[2]
            hist = np.empty((max_grey_value, dimensions))

            for dimension in range(0, dimensions):
                for gray_value in range(0, max_grey_value):
                    image_2d = image[:, :, dimension]
                    hist[gray_value, dimension] = len(image_2d[image_2d == gray_value])
        else:
            hist = np.empty((max_grey_value,))

            for gray_value in range(0, max_grey_value):
                hist[gray_value] = len(image[image == gray_value])

        return hist

    @staticmethod
    def _get_averaged_images(img, kernels):
        return np.array([signal.convolve2d(img, kernel, 'same') for kernel in kernels])

    @staticmethod
    def _get_average_values_for_every_pixel(img, number_kernels):
        """
        :param img: the image to be used in order to calculate averaged images
        :param number_kernels: number of kernels to be used in order to calculate the averaged images
        :return: averaged images with the shape:
                 (image height * image width, number averaged images)
                 Every row represents one pixel and its averaged values.
                 I. e. x[0] represents the first pixel and contains an array with k
                 averaged pixels where k are the number of used kernels.
        """
        kernels = ExactHistogramMatcher._kernel_mapping[number_kernels]
        averaged_images = ExactHistogramMatcher._get_averaged_images(img, kernels)
        img_size = averaged_images[0].shape[0] * averaged_images[0].shape[1]

        # shape of averaged_images: (number averaged images, height, width).
        # Reshape in a way, that one row contains all averaged values of pixel in position (x, y)
        reshaped_averaged_images = averaged_images.reshape((number_kernels, img_size))
        transposed_averaged_images = reshaped_averaged_images.transpose()
        return transposed_averaged_images

    @staticmethod
    def sort_rows_lexicographically(matrix):
        # Because lexsort in numpy sorts after the last row,
        # then after the second last row etc., we have to rotate
        # the matrix in order to sort all rows after the first column,
        # and then after the second column etc.

        rotated_matrix = np.rot90(matrix)

        # TODO lexsort is very memory hungry! If the image is too big, this can result in SIG 9!
        sorted_indices = np.lexsort(rotated_matrix)
        return matrix[sorted_indices]

    @staticmethod
    def _match_to_histogram(image, reference_histogram, number_kernels):
        """
        :param image: image as numpy array.
        :param reference_histogram: reference histogram as numpy array
        :param number_kernels: The more kernels you use in order to calculate average images,
                               the more likely it is, the resulting image will have the exact
                               histogram like the reference histogram
        :return: The image with the exact reference histogram.
        """
        img_size = image.shape[0] * image.shape[1]

        merged_images = np.empty((img_size, number_kernels + 2))

        # The first column are the original pixel values.
        merged_images[:, 0] = image.reshape((img_size,))

        # The last column of this array represents the flattened image indices.
        # These indices are necessary to keep track of the pixel positions
        # after they haven been sorted lexicographically according their values.
        indices_of_flattened_image = np.arange(img_size).transpose()
        merged_images[:, -1] = indices_of_flattened_image

        # Calculate average images and add them to merged_images
        averaged_images = ExactHistogramMatcher._get_average_values_for_every_pixel(image, number_kernels)
        for dimension in range(0, number_kernels):
            merged_images[:, dimension + 1] = averaged_images[:, dimension]

        # Sort the array according the original pixels values and then after
        # the average values of the respective pixel
        sorted_merged_images = ExactHistogramMatcher.sort_rows_lexicographically(merged_images)

        # Assign gray values according the distribution of the reference histogram
        index_start = 0
        for gray_value in range(0, len(reference_histogram)):
            index_end = int(index_start + reference_histogram[gray_value])
            sorted_merged_images[index_start:index_end, 0] = gray_value
            index_start = index_end

        # Sort back ordered by the flattened image index. The last column represents the index
        sorted_merged_images = sorted_merged_images[sorted_merged_images[:, -1].argsort()]
        new_target_img = sorted_merged_images[:, 0].reshape(image.shape)

        return new_target_img

    @staticmethod
    def match_image_to_histogram(image, reference_histogram, number_kernels=5):
        """
        :param image: image as numpy array.
        :param reference_histogram: reference histogram as numpy array
        :param number_kernels: The more kernels you use in order to calculate average images,
                               the more likely it is, the resulting image will have the exact
                               histogram like the reference histogram
        :return: The image with the exact reference histogram.
                 CAUTION: Don't save the image in a lossy format like JPEG,
                 because the compression algorithm will alter the histogram!
                 Use lossless formats like PNG.
        """
        if len(image.shape) == 3:
            # Image with more than one dimension. I. e. an RGB image.
            output = np.empty(image.shape)
            dimensions = image.shape[2]

            for dimension in range(0, dimensions):
                output[:, :, dimension] = ExactHistogramMatcher._match_to_histogram(image[:, :, dimension],
                                                                                    reference_histogram[:, dimension],
                                                                                    number_kernels)
        else:
            # Gray value image
            output = ExactHistogramMatcher._match_to_histogram(image,
                                                               reference_histogram,
                                                               number_kernels)

        return output
