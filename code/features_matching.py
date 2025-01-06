import cv2 as cv
import numpy as np

img_path1 = "images/image1.png"
img_path2 = "images/image2.png"


class FeatureMatcher():
    def read_and_resize(self, img_path, read_option, scale=1):
        img = cv.imread(img_path, read_option)
        
        img = cv.resize(img, None, fx=scale, fy=scale)

        return img

    def __init__(self, scale):
        self.scale = scale

    def SIFT(self, img_path):
        img = self.read_and_resize(img_path, cv.IMREAD_GRAYSCALE, scale=self.scale)

        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        image_with_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoints, descriptors, image_with_keypoints

    def get_original_image(self, img_path):
        img = self.read_and_resize(img_path, cv.IMREAD_COLOR, scale=self.scale)
        return img

    def get_grayscale_image(self, img_path):
        img = self.read_and_resize(img_path, cv.IMREAD_GRAYSCALE, scale=self.scale)
        return img

    def show_image(self, image):
        # show the combined image
        cv.imshow('Processed Image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_image(self, image, name):
        cv.imwrite(name, image)
        print("Image saved successfully!")


def match_features(img1_path, img2_path):
    fm = FeatureMatcher(scale=0.4)
    
    # detect and describe features
    keypoints1, descriptors1, _ = fm.SIFT(img1_path)
    keypoints2, descriptors2, _ = fm.SIFT(img2_path)

    img1 = fm.get_grayscale_image(img_path1)
    img2 = fm.get_grayscale_image(img_path2)
    
    # 1. Brute-Force Matcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches_bf = bf.match(descriptors1, descriptors2)
    matches_bf = sorted(matches_bf, key=lambda x: x.distance)
    
    # Draw matches ussing Brute-Force Matcher
    bf_matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches_bf[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fm.save_image(bf_matched_img, "output1.png")
    cv.imshow("Brute-Force Matching", bf_matched_img)

    # 2. FLANN-Based Matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN use K-D Tree
    search_params = dict(checks=50)  # Number of checks
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches_flann = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    
    # Draw maches from FLANN-Based Matcher
    all_matches_img = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches_flann[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fm.save_image(all_matches_img, "output2.png")
    cv.imshow("FLANN-Based Matching (All Matches)", all_matches_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def combine_and_save_image(image1, image2):
    # combine 2 images for comparision
    combined_image = np.concatenate((image1, image2), axis=1)

    fm.save_image(combined_image, "output.png")
    fm.show_image(combined_image)


def sift_feature_matching_with_filtering(img1_path, img2_path):
    fm = FeatureMatcher(scale=0.4)
    
    # Detect and describe features
    keypoints1, descriptors1, _ = fm.SIFT(img1_path)
    keypoints2, descriptors2, _ = fm.SIFT(img2_path)

    img1 = fm.get_grayscale_image(img_path1)
    img2 = fm.get_grayscale_image(img_path2)

    # Use FLANN to find feature matches
    index_params = dict(algorithm=1, trees=5)  # KD-Tree
    search_params = dict(checks=50)  # Number of checks
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract coordinates of good matches
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to find homography and filter outliers
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Draw matches after filtering
        filtered_matches = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i]]
        result_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, filtered_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        result_img = None
        print("Not enough good matches to apply RANSAC!")

    # Display the result
    if result_img is not None:
        cv.imshow('Filtered Matches', result_img)
        fm.save_image(result_img, "after_filter.png")
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Unable to display the result!")


    
if __name__ == '__main__':
    fm = FeatureMatcher(scale=0.4)

    keypoints, descriptor, image_with_keypoints = fm.SIFT(img_path1)
    # combine_and_save_image(fm.get_original_image(img_path1), fm.get_original_image(img_path2))

    # match_features(img_path1, img_path2)
    sift_feature_matching_with_filtering(img_path1, img_path2)


