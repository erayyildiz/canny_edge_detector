import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class EdgeDetector(object):
    # Sobel filters as numpy array
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Gaussian Filter for smoothening
    # Discrete approximation to Gaussian function(sigma = 1.0)
    gaussian_filter = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]], dtype=np.float)
    gaussian_filter = (1.0 / 273) * gaussian_filter

    def __init__(self):
        pass

    def _read_image(self, path):
        """
        Read an image from given path and return coressponded numpy array
        :param path: image path to be read
        :return:  a numpy array representing the image
        """
        img = np.asarray(Image.open(path))
        # If RGB image, convert to gray scale
        if len(img.shape) == 3:
            return self._rgb2gray(img)
        elif len(img.shape) == 2:
            return img
        else:
            raise IOError("Check the image in {}".format(path))

    def _rgb2gray(self, rgb):
        """
        convert given rgb image to grayscale image
        :param rgb: rgb image as numpy array
        :return: gray scale image as numpy array
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _add_padding(self, x, p):
        """
        Add padding to given matrix
        :param x: matrix (numpy array)
        :param p: padding size
        :return: padded matrix as numpy array
        """
        out = np.zeros((x.shape[0] + 2 * p, x.shape[1] + 2 * p))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i + p][j + p] = x[i][j]
        return out

    def _convolution(self, x, w, padding=0):
        """
        Apply convolution to given matrix
        :param x: matrix as numpy array
        :param w: kernel matrix as numpy array
        :param padding: padding size
        :return: output matrix as nupy array
        """
        img_height, img_width = x.shape
        filter_height, filter_width = w.shape

        # Apply Padding
        x_pad = self._add_padding(x, padding)


        # Calculate output matrix shape
        output_height = 1 + (img_height + 2 * padding - filter_height)
        output_width = 1 + (img_width + 2 * padding - filter_width)

        out = np.zeros((output_height, output_width))

        # Apply convolution
        for k in range(output_height):
            if k + filter_height > img_height:
                break
            for l in range(output_width):
                if l + filter_width > img_width:
                    break
                out[k, l] = np.sum(
                    x_pad[k:k + filter_height, l:l + filter_width] * w)
        return out


    def _non_maximum_suppression(self, gradient_magnitude, theta):
        """
        Apply non-maximum supression for given gradient magnitude and theta matrixes
        :param gradient_magnitude:
        :param theta:
        :return:
        """
        assert gradient_magnitude.shape == theta.shape
        for i in range(gradient_magnitude.shape[0]):
            for j in range(gradient_magnitude.shape[1]):
                if i == 0 or i == gradient_magnitude.shape[0] - 1 or j == 0 or j == gradient_magnitude.shape[1] - 1:
                    gradient_magnitude[i, j] = 0
                    continue
            direction = theta[i, j] % 4
            if direction == 0:  # horizontal +
                if gradient_magnitude[i, j] <= gradient_magnitude[i, j - 1] or gradient_magnitude[i, j] <= \
                        gradient_magnitude[i, j + 1]:
                    gradient_magnitude[i, j] = 0
                if direction == 1:  # horizontal -
                    if gradient_magnitude[i, j] <= gradient_magnitude[i - 1, j + 1] or gradient_magnitude[i, j] <= \
                            gradient_magnitude[i + 1, j - 1]:
                        gradient_magnitude[i, j] = 0
                if direction == 2:  # vertical +
                    if gradient_magnitude[i, j] <= gradient_magnitude[i - 1, j] or gradient_magnitude[i, j] <= \
                            gradient_magnitude[i + 1, j]:
                        gradient_magnitude[i, j] = 0
                if direction == 3:  # vertical -
                    if gradient_magnitude[i, j] <= gradient_magnitude[i - 1, j - 1] or gradient_magnitude[i, j] <= \
                            gradient_magnitude[i + 1, j + 1]:
                        gradient_magnitude[i, j] = 0

        return gradient_magnitude


    def _check_strong_in_neighbors(self, i, j, img, strong_value):
        # Check whether given pixel i,j has a strong neighbor or not
        neighbors = img[max([0, i-1]):min([img.shape[0],i+2]), max([0, j-1]):min([img.shape[1], j+2])]
        if strong_value in neighbors:
            return True


    def _hysteresis_threshold(self, x, low_threshold=60, high_threshold=120):
        """
        Apply Hysteresis Threshold to given matrix
        :param x: input matrix as numpy array
        :param low_threshold:
        :param high_threshold:
        :return:
        """
        strong_i, strong_j = np.where(x > high_threshold)
        candidate_i, candidate_j = np.where((x >= low_threshold) & (x <= high_threshold))
        zero_i, zero_j = np.where(x < low_threshold)
        strong = np.int32(255)
        candidate = np.int32(50)
        x[strong_i, strong_j] = strong
        x[candidate_i, candidate_j] = candidate
        x[zero_i, zero_j] = np.int32(0)

        M, N = x.shape
        for i in range(M):
            for j in range(N):
                if x[i, j] == candidate:
                    # check if one of the neighbours is strong
                    if self._check_strong_in_neighbors(i, j, x, strong):
                        x[i, j] = strong
                    else:
                        x[i, j] = 0
        return x

    def _show_img(self, img):
        """
        Plot given image
        :param img: Image to be plotted
        :return:
        """
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

    def _save_img(self, img_arr, path):
        """
        Save given image to file
        :param img_arr: Imput image
        :param path: Path to save image
        :return:
        """
        img = Image.fromarray(np.uint8(img_arr))
        img.save(path)

    def detect_edges(self, img_path, output_path, show_outputs_of_steps=False):
        """
        This method detects edges in input image using Canny Edge Detection Algortihm
        :param img_path: Path for input image
        :param output_path: Path for output image
        :param show_outputs_of_steps: if true, the program will plot the images that are generated in each step
        :return: Image with edges
        """

        img_data = self._read_image(img_path)
        if show_outputs_of_steps:
            self._show_img(img_data)
        # Gaussion filter for smoothining
        img_data_smoothened = self._convolution(img_data, EdgeDetector.gaussian_filter)
        # Find magnitude and angle of gradient
        gradient_magnitude_x = self._convolution(img_data_smoothened, EdgeDetector.sobel_filter_x, padding=0)
        if show_outputs_of_steps:
            self._show_img(gradient_magnitude_x)
        gradient_magnitude_y = self._convolution(img_data_smoothened, EdgeDetector.sobel_filter_y, padding=0)
        if show_outputs_of_steps:
            self._show_img(gradient_magnitude_y)
        gradient_magnitude = np.sqrt(np.square(gradient_magnitude_x) + np.square(gradient_magnitude_y))
        if show_outputs_of_steps:
            self._show_img(gradient_magnitude)
        theta = np.arctan2(gradient_magnitude_x, gradient_magnitude_x)

        # Non Maximum Suppression
        res = self._non_maximum_suppression(gradient_magnitude, theta)
        if show_outputs_of_steps:
            self._show_img(res)

        # Hysteresis Threshold
        res = self._hysteresis_threshold(res)
        if show_outputs_of_steps:
            self._show_img(res)
        self._save_img(res, output_path)
