import cv2
import numpy as np
import imutils
import os
import sys

def knn_training():
    img = cv2.imread("training.png")
    if img.shape[1] > 640:
        img = imutils.resize(img, width=640)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("original", img)
    cv2.waitKey(0)

    # Binary Threshold Image
    ret, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshold", thresh_img)
    cv2.waitKey(0)

    # Setup Thresholds
    img_width = 20
    img_height = 30
    area_threshold = 17.5

    # Setup Flattened Images Features
    flattened_img = np.empty((0, img_width * img_height))

    # Setup Character Classifications
    classification = []
    characters = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]


    # Find Contours
    thresh_img_clone = thresh_img.copy()
    cnts = cv2.findContours(thresh_img_clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[1]

    for contour in contours:
        if cv2.contourArea(contour) >= area_threshold:
            print(cv2.contourArea(contour))
            boxX, boxY, boxW, boxH = cv2.boundingRect(contour)

            # draw currently detected character
            cv2.rectangle(img, (boxX, boxY), (boxX+boxW, boxY+boxH), (255, 0, 0), 2)

            # Current detected character
            current_char = thresh_img[boxY:boxY+boxH, boxX:boxX+boxW]
            resized_cur_char = cv2.resize(current_char, (img_width, img_height))
            reshape_cur_char = resized_cur_char.reshape((1, img_width*img_height))
            flattened_img = np.append(flattened_img, reshape_cur_char, axis=0)

            cv2.imshow("Training Image", img)
            cv2.imshow("Current Character", resized_cur_char)

            # Get current character classification from keyboard input
            current_class = cv2.waitKey(0)
            classification.append(current_class)

    output_classification = np.array(classification, np.float32)
    output_classification = output_classification.reshape((output_classification.size, 1))

    np.savetxt("classifications.txt", output_classification)
    np.savetxt("flattenedImageFeatures.txt", flattened_img)

    print("Training Complete!")

    return

if __name__ == "__main__":
    knn_training()














