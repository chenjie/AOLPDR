import cv2
import numpy as np
import imutils

def detectPlates(car_image):
    # Since license plate have great contrast between plate background
    # and plate characters, we perform Black Hat operation on gray scale
    # aka the difference between source image and Closing operation in order
    # to enhance image

    img = cv2.imread(car_image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, rectangle_kernel)

    cv2.imshow("black hat", black_hat)
    cv2.waitKey(0)

    # Perform Closing operation (Dilation followed by Erosion)
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, square_kernel)
    cv2.imshow("closing", closing)
    cv2.waitKey(0)

    # Binary threshold
    ret, threshold = cv2.threshold(closing, 50, 255, cv2.THRESH_BINARY)

    cv2.imshow("binary threshold", threshold)
    cv2.waitKey(0)

    # Sobel
    ddepth = cv2.CV_32F
    sobel_x = cv2.Sobel(black_hat, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    cv2.imshow("sobel x direction", sobel_x)
    cv2.waitKey(0)














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

    detectPlates(car2)






