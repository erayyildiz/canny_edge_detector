import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class EdgeDetector(object):
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def __init__(self):
        pass

    def _read_image(self, path):
        img = mpimg.imread(path)
        gray = self._rgb2gray(img)
        img_data = np.array(gray)
        return img_data

    def _rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _add_padding(self, x, p):
        out = np.zeros((x.shape[0] + 2 * p, x.shape[1] + 2 * p))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i + p][j + p] = x[i][j]
        return out

    def _convolution(self, x, w, padding=0):
        img_height, img_width = x.shape
        filter_height, filter_width = w.shape

        x_pad = self._add_padding(x, padding)

        output_height = 1 + (img_height + 2 * padding - filter_height)
        output_width = 1 + (img_width + 2 * padding - filter_width)

        out = np.zeros((output_height, output_width))

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

    def _hysteresis_threshold(self, x, low_threshold=100, high_threshold=220):

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
                    # check if one of the neighbours is strong (=255 by default)
                    if ((x[i + 1, j] == strong) or (x[i - 1, j] == strong)
                        or (x[i, j + 1] == strong) or (x[i, j - 1] == strong)
                        or (x[i + 1, j + 1] == strong) or (x[i - 1, j - 1] == strong)):
                        x[i, j] = strong
                    else:
                        x[i, j] = 0
        return x

    def _show_img(self, img):
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

    def _save_img(self, img_arr, path):
        mpimg.imsave(path, img_arr, cmap=plt.get_cmap('gray'), format="jpg")


    def detect_edges(self, img_path, output_path, show_outputs_of_steps=False):
        img_data = self._read_image(img_path)
        if show_outputs_of_steps:
            self._show_img(img_data)

        # Find magnitude and angle of gradient
        gradient_magnitude_x = self._convolution(img_data, EdgeDetector.sobel_filter_x, padding=0)
        if show_outputs_of_steps:
            self._show_img(gradient_magnitude_x)
        gradient_magnitude_y = self._convolution(img_data, EdgeDetector.sobel_filter_y, padding=0)
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
