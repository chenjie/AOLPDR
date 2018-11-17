import cv2
import imutils
import numpy as np # For general purpose array manipulation
import scipy.fftpack # For FFT2
from skimage import measure
from skimage.filters import threshold_local
from skimage import segmentation

def segmentation(plate_image):
    plate_img = cv2.imread(plate_image)
    cvt_plate = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    thresh_local = threshold_local(cvt_plate, 29, offset=15, method="gaussian")
    threshold = (cvt_plate > thresh_local).astype("uint8") * 255
    threshold = cv2.bitwise_not(threshold)

    # resize the license plate region to a canonical size
    plate = imutils.resize(plate_img, width=400)
    thresh = imutils.resize(threshold, width=400)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

# segmentation('plates/plate1.png')


def threshold_plate(plate_image):
    plate = cv2.imread(plate_image)
    gray_plate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray_plate, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Thresh Binary Inverse", threshold)
    # cv2.waitKey(0)

    labels = measure.label(threshold, neighbors=8, background=0)
    charCandidates = np.zeros(threshold.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue


        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(threshold.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cv2.imshow("label", labelMask)
        cv2.waitKey(0)


# threshold_plate("plates/plate1.png")


def threshold_plate_enhance(plate_image):
    img = cv2.imread('plates/plate1.png', 0)

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 90
    Ithresh = 255 * Ithresh.astype("uint8")

    # Show all images

    cv2.imshow('Thresholded Result', Ithresh)
    cv2.waitKey(0)

# threshold_plate_enhance("plates/plate1.png")

if __name__ == "__main__":
    threshold_plate("plates/plate1.png")
