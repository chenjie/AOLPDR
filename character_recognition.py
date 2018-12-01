import cv2
import numpy as np
import character_segmentation as segmenation

def recognition(charCandidate, charThreshold):
    # Load Relevant Knn Files
    classification = np.loadtxt("classifications.txt", np.float32)
    flattened_image_features = np.loadtxt("flattenedImageFeatures.txt", np.float32)

    # Reshape
    classification = classification.reshape((classification.size, 1))

    # KNN Training
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(flattened_image_features, cv2.ml.ROW_SAMPLE, classification)

    # Find Contours
    cnts = cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    boxes = []

    # Get all boxes
    for c in cnts:
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))

    # Order boxes from left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Initialize Thresholds
    resize_w = 20
    resize_h = 30

    plate_number = ""

    for (startX, startY, endX, endY) in boxes:
        current_char = charThreshold[startY:endY, startX:endX]
        cv2.imshow("character", current_char)
        cv2.waitKey(0)

        cur_char_resize = cv2.resize(current_char, (resize_w, resize_h))
        cur_char_reshape = cur_char_resize.reshape((1, resize_w * resize_h))
        cur_feature = np.float32(cur_char_reshape)

        _, result, _, _ = kNearest.findNearest(cur_feature, k=1)

        plate_char = str(chr(int(result[0][0])))
        plate_number += plate_char

    print("Detection Completed:")
    print(plate_number)

    return


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
    plate11 = "plates/plate11.png"
    plate12 = "plates/plate12.png"
    plate13 = "plates/plate13.png"
    plate14 = "plates/plate14.png"

    charCandidate, charThreshold = segmenation.plate_segmentation(plate12)
    recognition(charCandidate, charThreshold)










