def calculate_threshold():
    aspect_ratio = [0.55, 0.3714, 0.6176, 0.6, 0.57, 0.416, 0.58, 0.611, 0.56, 0.59, 0.588, 0.611, 0.58, 0.62857, 0.6285, 0.6285, 0.611]
    height_ratio = [0.875, 0.45, 0.4375, 0.425, 0.4375, 0.43, 0.45, 0.45, 0.45, 0.46, 0.46, 0.46, 0.42, 0.45, 0.4375, 0.4375, 0.4375, 0.45]
    solidity = [0.026, 0.05493, 0.305, 0.2808, 0.2326, 0.45, 0.38148, 0.5462, 0.367, 0.32, 0.48, 0.5878, 0.1963, 0.85, 0.323, 0.5772, 0.3344, 0.875, 0.34217]

    print("aspect_ration max: {}".format(str(max(aspect_ratio))))
    print("aspect_ration min: {}".format(str(min(aspect_ratio))))
    print("height_ratio max: {}".format(str(max(height_ratio))))
    print("height_ratio min: {}".format(str(min(height_ratio))))
    print("solidity max: {}".format(str(max(solidity))))
    print("solidity min: {}".format(str(min(solidity))))

calculate_threshold()