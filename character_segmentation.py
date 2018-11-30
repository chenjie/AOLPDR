import cv2
import imutils
import numpy as np # For general purpose array manipulation
import scipy.fftpack # For FFT2
from skimage import measure
from skimage.filters import threshold_local
from skimage import segmentation

PLATE_CHAR_ASPECT_RATIO = 1.0
PLATE_CHAR_HEIGHT_RATIO_UPPER = 0.95
PLATE_CHAR_HEIGHT_RATIO_LOWER = 0.4


def segmentation(plate_image):
    plate = cv2.imread(plate_image)
    gray_plate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    # Transform plate to binary
    ret, threshold = cv2.threshold(gray_plate, 110, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresh Binary Inverse", threshold)
    cv2.waitKey(0)

    # Find connecting regions of threshold regions
    connecting_regions = measure.label(threshold, neighbors=8, background=0)
    unique_regions = np.unique(connecting_regions)
    charCandidates = np.zeros(threshold.shape, dtype="uint8")
    count = 0

    # loop over the unique components
    for region in unique_regions:
        # if this is the background label, ignore it
        if region == 0:
            continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        regionMask = np.zeros(threshold.shape, dtype="uint8")
        regionMask[connecting_regions == region] = 255
        cnts = cv2.findContours(regionMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("label", regionMask)
        # cv2.waitKey(0)

        cnts = cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:
            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solidity, and height ratio for the component
            aspectRatio = boxW / float(boxH)
            heightRatio = boxH / float(plate.shape[0])
            solidity = cv2.contourArea(c) / float(boxW * boxH)

            # print("aspectRatio: " + str(aspectRatio))
            # print("heightRatio: " + str(heightRatio))
            # print("solidity: " + str(solidity))
            # print("===================================")

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = 0.2 < aspectRatio < 0.46
            keepSolidity = solidity > 0.25
            keepHeight = heightRatio > 0.3 and heightRatio < 0.5

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                count += 1

    # cv2.imshow("charCandidates", charCandidates)
    # cv2.waitKey(0)
    print("There are: " + str(len(np.unique(connecting_regions))) + " connecting region")
    print(str(count) + " regions are plate characters")

    if count <= 5:
        print("Using enhance algorithm")

        threshold = threshold_plate_enhance(plate_image)
        cv2.imshow("Thresh Binary Inverse Enhance", threshold)
        cv2.waitKey(0)

        # Find connecting regions of threshold regions
        connecting_regions = measure.label(threshold, neighbors=8, background=0)
        unique_regions = np.unique(connecting_regions)
        charCandidates = np.zeros(threshold.shape, dtype="uint8")
        count = 0

        # loop over the unique components
        for region in unique_regions:
            # if this is the background label, ignore it
            if region == 0:
                continue

            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            regionMask = np.zeros(threshold.shape, dtype="uint8")
            regionMask[connecting_regions == region] = 255
            cnts = cv2.findContours(regionMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow("label", regionMask)
            # cv2.waitKey(0)

            cnts = cnts[1]

            # ensure at least one contour was found in the mask
            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                heightRatio = boxH / float(plate.shape[0])
                solidity = cv2.contourArea(c) / float(boxW * boxH)

                # print("aspectRatio: " + str(aspectRatio))
                # print("heightRatio: " + str(heightRatio))
                # print("solidity: " + str(solidity))
                # print("===================================")

                # determine if the aspect ratio, solidity, and height of the contour pass
                # the rules tests
                keepAspectRatio = 0.2 < aspectRatio < 0.46
                keepSolidity = solidity > 0.25
                keepHeight = heightRatio > 0.3 and heightRatio < 0.5

                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                    count += 1

        # cv2.imshow("charCandidates", charCandidates)
        # cv2.waitKey(0)
        print("There are: " + str(len(np.unique(connecting_regions))) + " connecting region")
        print(str(count) + " regions are plate characters")

    charThreshold = cv2.bitwise_and(threshold, threshold, mask=charCandidates)
    # cv2.imshow("charThreshold", charThreshold)
    # cv2.waitKey(0)

    return (charCandidates, charThreshold)



def threshold_plate_enhance(plate_image):
    img = cv2.imread(plate_image, 0)

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

    return Ithresh

def scissor(plate_image):
    charCandidate, charThreshold = segmentation(plate_image)

    cnts =  cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    boxes = []
    chars = []

    for c in cnts:
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))

    # Order boxes from left to right
    boxes = sorted(boxes, key=lambda b:b[0])

    for (startX, startY, endX, endY) in boxes:
        current_char = charThreshold[startY:endY, startX:endX]
        chars.append(current_char)
        cv2.imshow("character", current_char)
        cv2.waitKey(0)

    return chars


if __name__ == "__main__":
    plate1 = "plates/plate1.png"
    plate2 = "plates/plate2.jpeg"
    plate3 = "plates/plate3.png"
    plate4 = "plates/plate4.png"
    plate5 = "plates/plate5.png"
    plate6 = "plates/plate6.png"
    plate7 = "plates/plate7.png"
    plate8 = "plates/plate8.png"
    plate9 = "plates/plate9.png"
    plate10 = "plates/plate10.png"


    # segmentation(plate1)
    #threshold_plate_enhance(plate6)
    scissor(plate1)


