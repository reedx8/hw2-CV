# To run enter the following command in the terminal: python3 im*.py

import cv2
import sys
import numpy as np

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)

    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None

    # to be completed ...

    return best_H

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================

    # converting images from BGR color space to GRAY color space (algo only works with gray scale images?)
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # finding keypoints ("kp") in both img_1 and img_2
    kp_1 = sift.detect(gray_1, None)
    kp_2 = sift.detect(gray_2, None)
    
    # draws circles around each keypoint
    img_1 = cv2.drawKeypoints(gray_1, kp_1, img_1)
    img_2 = cv2.drawKeypoints(gray_2, kp_2, img_2)

    # img_#_feat are images containing circles around keypoints
    img_1_feat = cv2.drawKeypoints(gray_1, kp_1,img_1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_2_feat = cv2.drawKeypoints(gray_2, kp_2, img_2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imwrite('./samples/test_sift.jpg',img_1_feat)

    # cv2.imshow("image", img_1_feat)
    # cv2.waitKey(0)
    # cv2.imshow("image", img_2_feat)
    # cv2.waitKey(0)


    # gets a list of keypoints (kp_list) and numpy array of shape (des)
    kp_list_1, des_array_1 = sift.compute(gray_1, kp_1)
    kp_list_2, des_array_2 = sift.compute(gray_2, kp_2)


    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    bf = cv2.BFMatcher()
    matched_keypoints = bf.knnMatch(des_array_1, des_array_2, k=2)

    # Apply ratio test
    list_pairs_matched_keypoints = [] # good matches
    for m, n in matched_keypoints:
        if m.distance < ratio_robustness * n.distance:
            list_pairs_matched_keypoints.append([m, n])

    line_img = cv2.drawMatchesKnn(gray_1, kp_list_1, gray_2, kp_list_2, list_pairs_matched_keypoints, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("image", line_img)
    cv2.waitKey(0)

    return list_pairs_matched_keypoints

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...

    return img_panorama

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)


    return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW2: image stitching')
    print('==================================================')

    # path_file_image_1 = sys.argv[1]
    # path_file_image_2 = sys.argv[2]
    # path_file_image_result = sys.argv[3]

    path_file_image_1 = './samples/im3.jpg'
    path_file_image_2 = './samples/im4.jpg'
    # path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)
    # cv2.imshow("image", img_1)
    # cv2.waitKey(0)
    # cv2.imshow("image", img_2)
    # cv2.waitKey(0)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    # cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))

