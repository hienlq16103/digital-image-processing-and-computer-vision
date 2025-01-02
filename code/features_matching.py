import numpy as np
import cv2
import matplotlib.pyplot as plt

def brute_force():
    img1 = cv2.imread('images/image1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/image2.png', cv2.IMREAD_GRAYSCALE)
    
    scale_x = 0.4
    scale_y = 0.4
    img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y)
    img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x:x.distance)
    nMatches = 20
    imgMatch = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:nMatches],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def brute_force():
    img1 = cv2.imread('images/image1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/image2.png', cv2.IMREAD_GRAYSCALE)
    
    scale_x = 0.4
    scale_y = 0.4
    img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y)
    img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y)
    
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matched_img = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('Matches', matched_img)
    # cv2.imshow('Matches', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

brute_force()