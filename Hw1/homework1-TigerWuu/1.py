import sys
import numpy as np
import cv2 as cv
import cv3D as c3

# k = 80# select the top k points

#################################
##### Q1-1:Feature Matching #####
################################# 
def get_sift_correspondences(img1, img2):
    global k
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # print("type : ",type(kp1))
    # print("keypoint1 : ", kp1[0].pt)
    matcher = cv.BFMatcher() # brutal force matching
    matches = matcher.knnMatch(des1, des2, k=2) # find the top 2 nearest match(type : Dmatch)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # if th top two nearest point have big discrepency (smaller than 0.75 times the distance of the larger one), we add it to the good_matches
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance) # in the order of small to large

    if k > len(good_matches):  # avoid selecting points over the length of good_matches
        k = len(good_matches)

    good_matches = good_matches[0:k]
    # kp1_select = [kp1[m.queryIdx] for m in good_matches] # the keypoint of the first image (query)
    # kp2_select = [kp2[m.trainIdx] for m in good_matches] # the keypoint of the second image (train)

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches]) # the keypoint of the first image (query)
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches]) # the keypoint of the second image (train)
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # ??
    
    ## resize to easily to show
    # img_draw_match = cv.resize(img_draw_match, (1280, 720), interpolation=cv.INTER_AREA)
    ## show image
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    return points1, points2


if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    k = int(sys.argv[4])

    points1, points2 = get_sift_correspondences(img1, img2)
    H = c3.findHomographyDLT(points1, points2, k)
    H_norm = c3.findHomographyNDLT(points1, points2, k, method=0 , threshold=3, iteration=2000, numPick=4)
    H_norm_RAN = c3.findHomographyNDLT(points1, points2, k, method="RANSAC" , threshold=3, iteration=2000, numPick=4)
    # H_cv, _ = cv.findHomography(points1, points2,cv.RANSAC) 
    errorH = c3.reprojectionErrorEstimation(H, gt_correspondences)
    errorH_norm = c3.reprojectionErrorEstimation(H_norm, gt_correspondences)
    errorH_norm_RAN = c3.reprojectionErrorEstimation(H_norm_RAN, gt_correspondences)
    # errorH_cv = c3.reprojectionErrorEstimation(H_cv, gt_correspondences)
    print("H : ",H)
    print("H_norm : ",H_norm)
    print("H_norm_RAN : ",H_norm_RAN)
    # print("H_cv : ",H_cv)
    print("Error DLT : ",errorH)
    print("Error NDLT : ", errorH_norm)
    print("Error NDLT+RANSAC : ", errorH_norm_RAN)
    # print("error H_cv : ", errorH_cv)