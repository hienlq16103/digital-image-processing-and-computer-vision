import cv2 as cv
import numpy as np

def match_features(img1_path, img2_path):
    # Đọc hai ảnh
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    
    # Kiểm tra nếu ảnh không được tải thành công
    if img1 is None or img2 is None:
        print("Không thể đọc một trong hai ảnh.")
        return
    
    # Khởi tạo SIFT
    sift = cv.SIFT_create()
    
    # Phát hiện và tính toán các mô tả đặc trưng
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # 1. Brute-Force Matcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches_bf = bf.match(descriptors1, descriptors2)
    matches_bf = sorted(matches_bf, key=lambda x: x.distance)
    
    # Vẽ kết quả khớp bằng Brute-Force Matcher
    bf_matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches_bf[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Brute-Force Matching", bf_matched_img)

    # 2. FLANN-Based Matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN sử dụng K-D Tree
    search_params = dict(checks=50)  # Số lần kiểm tra
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches_flann = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lọc các khớp chất lượng cao bằng phương pháp Lowe's ratio test
    good_matches = []
    for m, n in matches_flann:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Vẽ kết quả khớp bằng FLANN-Based Matcher
    flann_matched_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("FLANN-Based Matching", flann_matched_img)

    # Chờ cho đến khi nhấn phím bất kỳ để đóng cửa sổ
    cv.waitKey(0)
    cv.destroyAllWindows()

# Ví dụ gọi hàm
match_features('images/image1.png', 'images/image2.png')