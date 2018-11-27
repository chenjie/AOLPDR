import cv2
import numpy as np
import imutils

minPlateW=90
minPlateH=30

def detect_plate_region(car_image):
    # Since license plate have great contrast between plate background
    # and plate characters, we perform Black Hat operation on gray scale
    # aka the difference between source image and Closing operation in order
    # to enhance image

    img = cv2.imread(car_image)
    if img.shape[1] > 640:
        img = imutils.resize(img, width=640)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))

    # Perform Blackhat operation
    black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, rectangle_kernel)

    cv2.imshow("black hat", black_hat)
    cv2.waitKey(0)

    # Perform Closing operation (Dilation followed by Erosion)
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, square_kernel)
    cv2.imshow("closing", closing)
    cv2.waitKey(0)

    # Binary threshold
    ret, light_mask = cv2.threshold(closing, 50, 255, cv2.THRESH_BINARY)

    cv2.imshow("binary threshold", light_mask)
    cv2.waitKey(0)

    # Sobel
    ddepth = cv2.CV_32F
    sobel_x = cv2.Sobel(black_hat, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    sobel_x_abs = np.absolute(sobel_x)
    (min_val, max_val) = (np.min(sobel_x_abs), np.max(sobel_x_abs))
    sobel_x_scale = (255 * ((sobel_x_abs - min_val) / (max_val - min_val))).astype("uint8")
    cv2.imshow("sobel x-direction scale", sobel_x_scale)
    cv2.waitKey(0)

    # Gaussian Filter
    sobel_x_gaussian = cv2.GaussianBlur(sobel_x_scale, (5, 5), 0)
    # Perform Closing operation (Dilation followed by Erosion)
    sobel_x_closing = cv2.morphologyEx(sobel_x_gaussian, cv2.MORPH_CLOSE, rectangle_kernel)
    cv2.imshow("sobel x-direction closing", sobel_x_closing)
    cv2.waitKey(0)
    # Perform Otsu thresholding
    ret, otsu_thresh = cv2.threshold(sobel_x_closing, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("OTSU threshold", otsu_thresh)
    cv2.waitKey(0)

    # Clean up image
    thresh = cv2.erode(otsu_thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("Clean up: Closing", otsu_thresh)
    cv2.waitKey(0)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light_mask)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    cv2.imshow("Clean up: mask", thresh)
    cv2.waitKey(0)


    # Contour
    regions = []
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # clone = img.copy()
    # cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
    #
    # cv2.imshow("Contour", clone)
    # cv2.waitKey(0)

    # loop over the contours
    for c in cnts:
        # grab the bounding box associated with the contour and compute the area and
        # aspect ratio
        (w, h) = cv2.boundingRect(c)[2:]
        aspectRatio = w / float(h)

        # print("aspectRatio: " + str(aspectRatio))
        # print("w: " + str(w))
        # print("h: " + str(h))
        #
        # clone = img.copy()
        # cv2.drawContours(clone, c, -1, (0, 255, 0), 2)
        #
        # cv2.imshow("Contour", clone)
        # cv2.waitKey(0)


        # compute the rotated bounding box of the region
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

        # ensure the aspect ratio, width, and height of the bounding box fall within
        # tolerable limits, then update the list of license plate regions
        if (aspectRatio > 1.5 and aspectRatio < 6) and h > minPlateH and w > minPlateW:
            print("added")
            regions.append(box)

    return regions



def detect_plate(car_image):
    img = cv2.imread(car_image)
    if img.shape[1] > 640:
        img = imutils.resize(img, width=640)
    regions = detect_plate_region(car_image)
    # regions = detect_origin(car_image)
    print(len(regions))

    for region in regions:
        lpBox = np.array(region).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(img, [lpBox], -1, (0, 255, 0), 2)

    cv2.imshow("Detection", img)
    cv2.waitKey(0)



def plate_region(car_image):
    pass




if __name__ == "__main__":
    car1 = "cars/car1.JPG"
    car2 = "cars/car2.JPG"
    car3 = "cars/car3.JPG"
    car4 = "cars/car4.JPG"
    car5 = "cars/car5.JPG"
    car6 = "cars/car6.JPG"
    car7 = "cars/car7.jpg"
    car8 = "cars/car8.jpg"
    car9 = "cars/car9.jpg"
    car10 = "cars/car10.jpg"


    detect_plate(car2)






