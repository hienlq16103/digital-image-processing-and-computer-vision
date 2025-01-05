import cv2 as cv
import numpy as np

img_path = "images/image1.png"

def SIFT():
    img_color = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    scale_x = 0.4
    scale_y = 0.4
    img = cv.resize(img, None, fx=scale_x, fy=scale_y)
    img_color = cv.resize(img_color, None, fx=scale_x, fy=scale_y)
    
    
    sift = cv.SIFT_create()
    keypoints = sift.detect(img, None)
    image_with_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    combined_image = np.concatenate((img_color, image_with_keypoints), axis=1) 
    cv.imshow('Original and SIFT Keypoints', combined_image)

    # Chờ cho đến khi nhấn phím bất kỳ để đóng cửa sổ
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
SIFT()