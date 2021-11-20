import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import skimage.external.tifffile as tiffreader


class LinearNorm:
    def __init__(self, lower=0.005, upper=0.995):
        """
        Linearly normalize the input. The outliers are reset as the boundary values
        :param lower: The lowest image intensity
        :param upper: The highest image intensity
        """
        self.lower = lower
        self.upper = upper

    @staticmethod
    def intensity_bounds(intensity_dist, lower=0.05, upper=0.95):
        """
        Calculate the boundaries of the intensity distribution
        :param intensity_dist: Intensity distribution
        :param lower: The lower percentage
        :param upper: The upper percentage
        :return: inten_min, inten_max
        """
        per = intensity_dist[0] / np.sum(intensity_dist[0])
        cum_per = np.cumsum(per)
        inten_min = intensity_dist[1][np.where(cum_per >= lower)[0][0]]
        inten_max = intensity_dist[1][np.where(cum_per > upper)[0][0]]
        return inten_min, inten_max

    def forward(self, image):
        """
        Process the linear norm
        :param image: Input image array
        :return: Linearly normalized image
        """
        image_min = math.floor(np.min(image))
        image_max = math.ceil(np.max(image)) + 1
        intenHist = np.histogram(image, bins=np.linspace(image_min, image_max, image_max - image_min + 1))
        inten_min, inten_max = self.intensity_bounds(intenHist, lower=self.lower, upper=self.upper)
        # Normalize illumination
        image[image < inten_min] = inten_min
        image[image > inten_max] = inten_max
        return image


class CLAHENorm:
    def __init__(self, clipLimit):
        """
        Process CLAHE with specific clip limit
        :param clipLimit: Clip limit
        """
        self.clipLimit = clipLimit
        self.method = self.clahe

    @staticmethod
    def clahe(image, clipLimit):
        """
        Build the histogram equalization method
        :param image: Input image array
        :param clipLimit: THe clip limit
        :return: CLAHEed image
        """
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(16, 16))
        image = clahe.apply(image)
        return image

    def forward(self, image):
        """
        Process the CLAHE
        :param image: Input image array
        :return: CLAHEed image
        """
        return self.method(image, self.clipLimit)


class NormalizeTif:
    def __init__(self, resize_by=1.0, crop_size=None, norm_range=None, load_method=0, norm_method=0, lower=0.005, upper=0.995,
                 clipLimit=40.0):
        """
        Process the normalization with specified method
        :param resize_by: A ratio to resize the image
        :param crop_size: The crop size
        :param norm_range: The range for normalization
        :param load_method: 0 - tiffread; 1 - cv2
        :param norm_method: 0 - intensity cutoff; 1 - histogram equalization; 2 - CLAHE; -1 - return to input
        :param lower: The lower intensity cutoff
        :param upper: The upper intensity cutoff
        :param clipLimit: The clip limit for CLAHE
        """
        self.resize_by = resize_by
        self.crop_size = crop_size
        if norm_method == 1:
            self.method = self.histogram_equalization
        elif norm_method == 2:
            method = CLAHENorm(clipLimit=clipLimit)
            self.method = method.forward
        elif norm_method == -1:
            self.method = self.return_to_input
        else:
            method = LinearNorm(lower=lower, upper=upper)
            self.method = method.forward

        self.norm_range = norm_range
        self.load_method = load_method
        self.clipLimit = clipLimit

    @staticmethod
    def return_to_input(image):
        return image

    @staticmethod
    def histogram_equalization(image):
        """
        Process histogram equalization on an int16 image array
        :param image: Input image array
        :return: Histogram equalized int16 image array
        """
        hist, bins = np.histogram(image.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        # cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')
        image2 = cdf[image]
        return image2

    @staticmethod
    def histogram_equalization_cv2(image):
        """
        Use the build-in histogram equalization method in cv2, which is specified for int8 images.
        :param image: Input int8 array
        :return: Histogram equalized int8 image array
        """
        return cv2.equalizeHist(image)

    @staticmethod
    def image_load(file_name, mode):
        """
        Load the image
        :param file_name: The image file name
        :param mode: The loading norm_method: 0 - tiffreader; 1 - cv2 (int8)
        :return: Image array
        """
        if mode == 0:
            image = tiffreader.imread(file_name)
        else:
            image = cv2.imread(file_name, 0)
        return image

    @staticmethod
    def image_resize(image, resize_by):
        """
        Resize the image
        :param image: Input image array
        :param resize_by: The resize ratio
        :return: Resized image
        """
        if resize_by == 1:
            return image
        image = Image.fromarray(image).convert("I")
        img_size = np.array(image.size) * resize_by
        image = image.resize(img_size.astype(int), Image.ANTIALIAS)
        image = np.array(image, dtype=np.uint16)
        return image

    @staticmethod
    def image_norm(image, norm_range):
        """
        Normalize the image to norm_range
        :param image: Input image array
        :param norm_range: The range of output
        :return: Normalized image
        """
        if norm_range is None:
            return image
        image_min = math.floor(np.min(image))
        image_max = math.ceil(np.max(image))
        image = (image - image_min) / (image_max - image_min)
        image = image * (norm_range[1] - norm_range[0]) + norm_range[0]
        # Throw outliers
        image[image < norm_range[0]] = norm_range[0]
        image[image > norm_range[1]] = norm_range[1]
        return np.array(image, dtype=np.float32)

    @staticmethod
    def image_crop(image, crop_size):
        """
        Crop the image
        :param image: Input image array
        :param crop_size: The crop size
        :return: Cropped image
        """
        if crop_size:
            image_size = np.shape(image)
            if isinstance(crop_size, int):
                crop_size = [crop_size, crop_size]
            if crop_size[0] > image_size[0]:
                crop_size[0] = image_size[0]
            if crop_size[1] > image_size[1]:
                crop_size[1] = image_size[1]
            image = image[:crop_size[0], :crop_size[1]]
        return image

    def forward(self, file_name):
        """
        Process the normalization with the class specified method
        :param file_name: The name of a tif file or an image array
        :return: processed image
        """
        if isinstance(file_name, str):
            image = self.image_load(file_name, self.load_method)
        else:
            image = file_name
        image = self.image_crop(image, self.crop_size)
        image = self.image_resize(image, self.resize_by)
        image = self.method(image)
        image = self.image_norm(image, self.norm_range)
        return image

'''
def test():
    file_n = "/mnt/data/MyxoMovie/phase_tdTomato_gfp/24Hour_tdTomato_movies/LS3908_24hr_7.7.20/tdTomato/image_1445.tif"
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    m1 = NormalizeTif(resize_by=1.0, crop_size=None, norm_range=[0, 1], norm_method=0, lower=0.005, upper=0.995)
    m2 = NormalizeTif(resize_by=1.0, crop_size=None, norm_range=[0, 1], norm_method=1)
    m3 = NormalizeTif(resize_by=1.0, crop_size=None, norm_range=[0, 1], norm_method=2, clipLimit=2.0, load_method=1)
    cl1 = 4.0
    cl2 = 40.0
    m4 = NormalizeTif(resize_by=1.0, crop_size=None, norm_range=[0, 1], norm_method=2, clipLimit=cl1)
    m5 = NormalizeTif(resize_by=1.0, crop_size=None, norm_range=[0, 1], norm_method=2, clipLimit=cl2)
    image0 = tiffreader.imread(file_n)
    ax[0, 0].imshow(image0, cmap="gray")
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Original")
    image1 = m1.forward(file_n)
    ax[0, 1].imshow(image1, cmap="gray")
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Linear")
    image1 = m2.forward(file_n)
    ax[0, 2].imshow(image1, cmap="gray")
    ax[0, 2].axis("off")
    ax[0, 2].set_title("Histogram Equalization")
    image1 = m3.forward(file_n)
    ax[1, 0].imshow(image1, cmap="gray")
    ax[1, 0].axis("off")
    ax[1, 0].set_title("CLAHE uint8")
    image1 = m4.forward(file_n)
    ax[1, 1].imshow(image1, cmap="gray")
    ax[1, 1].axis("off")
    ax[1, 1].set_title("CLAHE uint16 clipLimit %d" % cl1)
    image1 = m5.forward(file_n)
    ax[1, 2].imshow(image1, cmap="gray")
    ax[1, 2].axis("off")
    ax[1, 2].set_title("CLAHE uint16 clipLimit %d" % cl2)
    plt.show()
    plt.close()
    n_bins = 100
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    image0 = tiffreader.imread(file_n)
    ax[0, 0].hist(image0.flatten(), n_bins, density=True)
    ax1 = ax[0, 0].twinx()
    ax1.axes.yaxis.set_ticklabels([])
    ax1.hist(image0.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange")
    ax[0, 0].set_title("Original")
    image1 = m1.forward(file_n)
    ax[0, 1].hist(image1.flatten(), n_bins, density=True)
    ax2 = ax[0, 1].twinx()
    ax2.axes.yaxis.set_ticklabels([])
    ax2.hist(image1.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange")
    ax[0, 1].set_title("Linear")
    image1 = m2.forward(file_n)
    ax[0, 2].hist(image1.flatten(), n_bins, density=True)
    ax3 = ax[0, 2].twinx()
    ax3.hist(image1.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange")
    ax[0, 2].set_title("Histogram Equalization")

    image1 = m3.forward(file_n)
    ax[1, 0].hist(image1.flatten(), n_bins, density=True)
    ax4 = ax[1, 0].twinx()
    ax4.axes.yaxis.set_ticklabels([])
    ax4.hist(image1.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange")
    ax[1, 0].set_title("CLAHE uint8")
    image1 = m4.forward(file_n)
    ax[1, 1].hist(image1.flatten(), n_bins, density=True)
    ax5 = ax[1, 1].twinx()
    ax5.axes.yaxis.set_ticklabels([])
    ax5.hist(image1.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange")
    ax[1, 1].set_title("CLAHE uint16 clipLimit %d" % cl1)
    image1 = m5.forward(file_n)
    ax[1, 2].hist(image1.flatten(), n_bins, density=True, label="histogram")
    ax6 = ax[1, 2].twinx()
    ax6.hist(image1.flatten(), n_bins, density=True, histtype="step", cumulative=True, color="tab:orange", label="cumulative")
    ax[1, 2].set_title("CLAHE uint16 clipLimit %d" % cl2)
    fig.legend(loc=(.8, .08))
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

test()
'''