import numpy as np
import cv2 as cv


########################################
##### Q1-2:Direct Linear Transform #####
######################################## 
def findHomographyDLT(points1, points2, k):
    # Construct array A from key points pairs 
    for i in range(k):
        u = points1[i][0] 
        v = points1[i][1] 
        u_p = points2[i][0] 
        v_p = points2[i][1] 
        A_new = np.array([[u, v, 1, 0, 0, 0, -u_p*u, -u_p*v, -u_p],
                          [0, 0, 0, u, v, 1, -v_p*u, -v_p*v, -v_p]])
        if i==0:
            A = A_new
        else:
            A = np.concatenate((A, A_new), axis=0)
    # apply SVD to A to get h
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    # print("U",U.shape)
    # print("S",S.shape)
    # print("Vh",Vh) 

    # reconstruct the homography from Vh
    h = Vh[-1]
    H = h.reshape((3,3))
    # print(H)
    return H

####################################################
##### Q1-3:Normailized Direct Linear Transform #####
#################################################### 
def findHomographyNDLT(points1, points2, k, method = 0, threshold = 0, iteration = 0, numPick = 0):
    # Construct array A from key points pairs 
    # translate the centroid to the origin
    if method == 0:
        centroid1 = 0
        centroid2 = 0
        scale1 = 0
        scale2 = 0
        for i in range(k):
            centroid1 += (points1[i]/k) 
            centroid2 += (points2[i]/k) 
        translatePoints1 = points1 - centroid1 
        translatePoints2 = points2 - centroid2
        for i in range(k):
            scale1 += (np.linalg.norm(translatePoints1[i])/k)
            scale2 += (np.linalg.norm(translatePoints2[i])/k)
    
        norTranslatePoints1 = translatePoints1* 2**0.5 / scale1
        norTranslatePoints2 = translatePoints2* 2**0.5 / scale2
    
        for i in range(k):  
            u = norTranslatePoints1[i][0] 
            v = norTranslatePoints1[i][1] 
            u_p = norTranslatePoints2[i][0] 
            v_p = norTranslatePoints2[i][1] 
            A_new = np.array([[u, v, 1, 0, 0, 0, -u_p*u, -u_p*v, -u_p],
                              [0, 0, 0, u, v, 1, -v_p*u, -v_p*v, -v_p]])
            if i==0:
                A = A_new
            else:
                A = np.concatenate((A, A_new), axis=0)
        # apply SVD to A to get h
        U, S, Vh = np.linalg.svd(A, full_matrices=True)

        # reconstruct the homography from Vh
        h_hat = Vh[-1]
        H_hat = h_hat.reshape((3,3))
        s1 = 2**0.5 /scale1
        s2 = 2**0.5 /scale2
        T = np.array([[s1,0,-s1*centroid1[0]],[0,s1,-s1*centroid1[1]],[0,0,1]])
        T_prime = np.array([[s2,0,-s2*centroid2[0]],[0,s2,-s2*centroid2[1]],[0,0,1]])
        H = np.dot(np.dot(np.linalg.inv(T_prime),H_hat),T)
    elif method == "RANSAC":
        pts1 = np.zeros([numPick,2])
        pts2 = np.zeros([numPick,2])
        points1Homo = np.concatenate((points1, np.ones((k,1))), axis=1)
        points2Homo = np.concatenate((points2, np.ones((k,1))), axis=1)
        bestNumInliers = 0
        
        for _ in range(iteration):
            randomlist = np.random.choice(range(k),size=numPick,replace=False)
            # print(randomlist)
            for i in range(numPick):
                pts1[i] = points1[randomlist[i]]
                pts2[i] = points2[randomlist[i]]
            
            centroid1 = 0
            centroid2 = 0
            scale1 = 0
            scale2 = 0
            for i in range(numPick):
                centroid1 += (pts1[i]/numPick) 
                centroid2 += (pts2[i]/numPick) 
            translatePoints1 = pts1 - centroid1 
            translatePoints2 = pts2 - centroid2
            for i in range(numPick):
                scale1 += (np.linalg.norm(translatePoints1[i])/numPick)
                scale2 += (np.linalg.norm(translatePoints2[i])/numPick)
    
            norTranslatePoints1 = translatePoints1* 2**0.5 / scale1
            norTranslatePoints2 = translatePoints2* 2**0.5 / scale2
    
            for i in range(numPick):  
                u = norTranslatePoints1[i][0] 
                v = norTranslatePoints1[i][1] 
                u_p = norTranslatePoints2[i][0] 
                v_p = norTranslatePoints2[i][1] 
                A_new = np.array([[u, v, 1, 0, 0, 0, -u_p*u, -u_p*v, -u_p],
                                  [0, 0, 0, u, v, 1, -v_p*u, -v_p*v, -v_p]])
                if i==0:
                    A = A_new
                else:
                    A = np.concatenate((A, A_new), axis=0)
            # apply SVD to A to get h
            U, S, Vh = np.linalg.svd(A, full_matrices=True)

            # reconstruct the homography from Vh
            h_hat = Vh[-1]
            H_hat = h_hat.reshape((3,3))
            s1 = 2**0.5 /scale1
            s2 = 2**0.5 /scale2
            T = np.array([[s1,0,-s1*centroid1[0]],[0,s1,-s1*centroid1[1]],[0,0,1]])
            T_prime = np.array([[s2,0,-s2*centroid2[0]],[0,s2,-s2*centroid2[1]],[0,0,1]])
            tempH = np.dot(np.dot(np.linalg.inv(T_prime),H_hat),T) # H_hat -> H by (T')^(-1)H_T

            # calculate the number of inliers
            numInliers = 0
            for i in range(k):
                pts2Homo_hat = np.dot(tempH,points1Homo[i])
                pts2Homo_hat = pts2Homo_hat/pts2Homo_hat[2]
                err =np.linalg.norm(pts2Homo_hat-points2Homo[i]) 
                if err < threshold:
                    numInliers += 1 
            # print(numInliers)
            if numInliers > bestNumInliers:
                bestNumInliers = numInliers
                print("best : ", bestNumInliers)
                H = tempH
        
    return H

def reprojectionErrorEstimation(H, gt):
    num = gt.shape[1] # number of data
    pts1_gt = gt[0]
    pts1_gt = np.concatenate((pts1_gt, np.ones((num,1))), axis=1)
    pts2_gt = gt[1]
    pts2_gt = np.concatenate((pts2_gt, np.ones((num,1))), axis=1)
    error = 0
    for i in range(num):
        pts2_hat = np.dot(H,pts1_gt[i])
        pts2_hat /= pts2_hat[2] # rescale after mutiply by scaling factor
        # print(np.dot(H,pts1_gt[i]))
        error += (np.linalg.norm(pts2_hat-pts2_gt[i]))/num 

    return error

###########################################
##### Q2-2:Warp Perspective Transform #####
###########################################
def warpPerspective(H , srcimage, dstimage, method = "NN"):
    invH = np.linalg.inv(H)
    A = 1 # for bilinear interpolation 

    for u in range(dstimage.shape[1]):
        for v in range(dstimage.shape[0]):
            dstimg_coord = np.array([u,v,1])  
            # print(dstimg_coord)
            srcimg_coord = np.dot(invH , dstimg_coord)
            srcimg_coord /= srcimg_coord[2] # rescale after mutiply by scaling factor
            if method == "NN":
                dstimage[v][u] = srcimage[int(srcimg_coord[1])][int(srcimg_coord[0])] ## backward warping
            elif method == "Bilinear":
                a_p = srcimg_coord[0] - int(srcimg_coord[0])
                b_p = srcimg_coord[1] - int(srcimg_coord[1])
                a1 = a_p*b_p
                a2 = (1-a_p)*b_p
                a3 = (1-a_p)*(1-b_p)
                a4 = a_p*(1-b_p)
                Q1 = [int(srcimg_coord[0]),int(srcimg_coord[1])]
                Q2 = [int(srcimg_coord[0])+1,int(srcimg_coord[1])]
                Q3 = [int(srcimg_coord[0])+1,int(srcimg_coord[1])+1]
                Q4 = [int(srcimg_coord[0]),int(srcimg_coord[1])+1]
                dstimage[v][u] = a3/A*srcimage[Q1[1]][Q1[0]]+a4/A*srcimage[Q2[1]][Q2[0]]+a1/A*srcimage[Q3[1]][Q3[0]]+a2/A*srcimage[Q4[1]][Q4[0]]
    return dstimage 
