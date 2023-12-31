import numpy as np
import cv2 as cv


############################################
##### HW1:Q1-2:Direct Linear Transform #####
############################################ 
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

########################################################
##### HW1:Q1-3:Normailized Direct Linear Transform #####
######################################################## 
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

###############################################
##### HW1:Q2-2:Warp Perspective Transform #####
###############################################
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

################################
##### HW2:Q1-1: P3P+RANSAC #####
################################
class poseEstimator:
    def __init__(self,cameraMatrix,distCoeffs) -> None:
        self.cameraMatrix= cameraMatrix
        self.distCoeffs = distCoeffs
    
    def featureMatching(self, query, model):
        kp_query, desc_query = query # xy(image plane), descriptor of points in index image (ex.200)
        kp_model, desc_model = model # XYZ(world plane), average_descriptor(2D) of points from 1 - 129081

        bf = cv.BFMatcher()
        matches = bf.knnMatch(desc_query,desc_model,k=2)

        gmatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                gmatches.append(m)

        points2Ds = np.empty((0,2))
        points3Ds = np.empty((0,3))

        for mat in gmatches:
            query_idx = mat.queryIdx
            model_idx = mat.trainIdx
            points2Ds = np.vstack((points2Ds,kp_query[query_idx]))
            points3Ds = np.vstack((points3Ds,kp_model[model_idx]))
        
        # undistort the 2D points         
        for i in range(len(points2Ds)): 
            points2Ds[i] = cv.undistortPoints(points2Ds[i].reshape((1,2)), self.cameraMatrix, self.distCoeffs, P=self.cameraMatrix)

        return points2Ds, points3Ds
    
    def solveP3PRANSAC(self, points2Ds, points3Ds, iterations=None, threshold = 5, confidence = 0.95, s=0.7):
        # P3P + RANSAC
        bestInliers = 0
        if iterations == None:
            iterations = int(np.log(1-confidence)/np.log(1-(1-s)**3)) # assume there are 70% outliers

        for _ in range(iterations):
            randomPoints = np.random.choice(range(len(points2Ds)), size=3, replace=False)
            points2D = np.concatenate((points2Ds[randomPoints[0]].reshape((1,2)), points2Ds[randomPoints[1]].reshape((1,2)), points2Ds[randomPoints[2]].reshape((1,2))), axis=0)
            points3D = np.concatenate((points3Ds[randomPoints[0]].reshape((1,3)), points3Ds[randomPoints[1]].reshape((1,3)), points3Ds[randomPoints[2]].reshape((1,3))), axis=0)
            retval, Rot_temp, Trans_temp = self.solveP3P(points2D, points3D)
            if retval == 0:
                continue

            inliers = 0
            
            for i in range(len(points3Ds)):
                u = np.dot(np.dot(self.cameraMatrix,Rot_temp),(points3Ds[i].reshape((3,1))-Trans_temp))
                u = u / u[2][0]
                err = self.__reprojectionError([u[0][0],u[1][0]], points2Ds[i])
                if err < threshold:
                    inliers += 1
            
            if inliers > bestInliers:
                bestInliers = inliers
                Rot_best, Trans_best = Rot_temp, Trans_temp
        Rot_best = np.real(Rot_best)
        return Rot_best, Trans_best 

    def solveP3P(self, points2D, points3D):
        # convert point2D from euclidean coordinate into homogeneous coordinate
        points2D = np.concatenate((points2D, np.ones((3,1))),axis=1)
        # definition
        v1 = np.dot(np.linalg.inv(self.cameraMatrix),points2D[0].reshape((3,1))) # points2D[0].shape = (3,)
        v2 = np.dot(np.linalg.inv(self.cameraMatrix),points2D[1].reshape((3,1)))
        v3 = np.dot(np.linalg.inv(self.cameraMatrix),points2D[2].reshape((3,1)))
        Cab = np.vdot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        Cbc = np.vdot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3))
        Cca = np.vdot(v3,v1)/(np.linalg.norm(v3)*np.linalg.norm(v1))
        Rab = np.linalg.norm(points3D[0]-points3D[1])
        Rbc = np.linalg.norm(points3D[1]-points3D[2])
        Rca = np.linalg.norm(points3D[2]-points3D[0])
        if Rab==0 or Rbc==0 or Rca==0:
            R = None
            T = None
            retval = 0
            return retval, R, T
        K1 = (Rbc/Rca)**2
        K2 = (Rbc/Rab)**2
        G0 = (K1*K2+K1-K2)**2- 4*(K1**2)*K2*(Cca**2)
        G1 = 4*(K1*K2+K1-K2)*K2*(1-K1)*Cab+ 4*K1*((K1*K2-K1+K2)*Cca*Cbc+2*K1*K2*Cab*Cca**2)
        G2 = (2*K2*(1-K1)*Cab)**2+ 2*(K1*K2-K1-K2)*(K1*K2+K1-K2)+ 4*K1*((K1-K2)*Cbc**2+K1*(1-K2)*Cca**2-2*(1+K1)*K2*Cab*Cca*Cbc)
        G3 = 4*(K1*K2-K1-K2)*K2*(1-K1)*Cab+ 4*K1*((K1*K2-K1+K2)*Cca*Cbc+2*K2*Cab*Cbc**2)
        G4 = (K1*K2-K1-K2)**2- 4*K1*K2*(Cbc**2)
        ComMat = np.array([[0, 0, 0, -G0/G4],
                           [1, 0, 0, -G1/G4],
                           [0, 1, 0, -G2/G4],
                           [0, 0, 1, -G3/G4]])
        # calculate x from the companion matrix of the fourth-order polynomial
        x, _ = np.linalg.eig(ComMat)
        # print(x)
        # x constraints pick one out of 4 solutions
        x = x[0] # for test
        # utilizing x to calculate a
        a = (Rab**2/(1+x**2-2*x*Cab))**0.5
        # utilizing a to calculate y
        y = (-(x**2-K1)+(1-K1)*(x**2*(1-K2)+2*x*K2*Cab-K2))/(2*(K1*Cca-x*Cbc)+2*x*Cbc*(1-K1))
        # utilizing a,x to calculate b
        b = x*a
        # utilizing a,y to calculate c
        c = y*a
        # utilizing a,b,c,x1,x2,x3 to calculate T
        # 
        x123 = np.cross((points3D[0]-points3D[1]).reshape((1,3)), (points3D[2]-points3D[1]).reshape((1,3)))
        
        A = np.array([[-1/2*(a**2-b**2-np.linalg.norm(points3D[0])**2+np.linalg.norm(points3D[1])**2)],
                      [-1/2*(a**2-c**2-np.linalg.norm(points3D[0])**2+np.linalg.norm(points3D[2])**2)],
                      [np.vdot(x123, points3D[0].reshape((1,3)))]])
        
        B = np.array([[points3D[0][0]-points3D[1][0], points3D[0][1]-points3D[1][1], points3D[0][2]-points3D[1][2]],
                      [points3D[0][0]-points3D[2][0], points3D[0][1]-points3D[2][1], points3D[0][2]-points3D[2][2]],
                      [x123[0][0], x123[0][1], x123[0][2]]]) 
        O = np.dot(np.linalg.inv(B),A)
        T1 = O + (a**2-np.linalg.norm(points3D[0].reshape((3,1))-O)**2)**0.5/np.linalg.norm(x123)*(x123.reshape((3,1)))  
        T2 = O - (a**2-np.linalg.norm(points3D[0].reshape((3,1))-O)**2)**0.5/np.linalg.norm(x123)*(x123.reshape((3,1)))  
        # print("test : ",(a**2-np.linalg.norm(points3D[0].reshape((3,1))-O)**2))
        # print("3D :", points3D[0])
        # print("O :", O)
        # print("test : ", (points3D[0].reshape((3,1))-O))
        # print("a : ", a) # might be complex ?
        Tc = np.concatenate((T1,T2), axis=1)
 
        # T constraints pick one out of 2 solutions
        for i in range(len(Tc[0])):
            T = Tc[:,i].reshape((3,1)) # for test

            # utilizing T,xi,vi to calculate lambdai, i = {1,2,3}
            lambda1 = np.linalg.norm(points3D[0].reshape((3,1))-T)/np.linalg.norm(v1)
            lambda2 = np.linalg.norm(points3D[1].reshape((3,1))-T)/np.linalg.norm(v2)
            lambda3 = np.linalg.norm(points3D[2].reshape((3,1))-T)/np.linalg.norm(v3)
            # utilizing T,xi,vi,lambdai to calculate R
            C = np.concatenate((lambda1*v1,lambda2*v2,lambda3*v3),axis=1)
            D = np.concatenate((points3D[0].reshape((3,1))-T, points3D[1].reshape((3,1))-T, points3D[2].reshape((3,1))-T),axis=1)
            R = np.dot(C,np.linalg.inv(D))
            if np.linalg.det(R) > 0:
                break
        retval = 1
        return retval, R,T
    
    def __reprojectionError(self, u, v):
        error = np.linalg.norm(u-v)
        return error

## Transform between world, camera, plane and homogeneous coordinates ## 
class coordinateTransform:
    def __init__(self, translation, rotation, intrinsic=None, world=None, camera=None, image=None) -> None:
        ### TO-DO
        self.translation_wc = translation
        self.rotation_wc = rotation
        self.intrinsic = intrinsic
        self.world = world
        self.camera = camera
        self.image = image
        
    def world2Camera(self):
        self.camera = np.dot(self.rotation_wc, self.world) + self.translation_wc
        return self.camera
    def camera2World(self):
        self.world = np.dot(self.rotWC2CW(self.rotation_wc), self.camera - self.translation_wc)
        return self.world
    def eucliden2Projective(self):
        pass
    def projecive2Euclidean(self):
        pass
    def transWC2CW(self, trans_wc):
        self.translation_cw = np.dot(-self.rotWC2CW(self.rotation_wc),self.translation_wc)
        return self.translation_cw
    def rotWC2CW(self, rot_wc):
        self.rotation_cw = np.linalg.inv(rot_wc)
        return self.rotation_cw
    

#########################
##### Visualization #####
#########################

class Point():
    def __init__(self, color, xyz) -> None:
        self.color = color
        self.xyz_world = np.array(xyz).reshape((3,1))
        self.xyz_camera = None
        self.uv_homo = None

def generate_cube_points(points_num, init_point, size, colors):
    points = []
    offset = size/points_num
    init_points = [init_point,
                   np.sum([init_point,[size, size, 0]],axis = 0).tolist(), 
                   np.sum([init_point,[size, 0, size]],axis = 0).tolist(),
                   np.sum([init_point,[0, 0, size]],axis = 0).tolist(),
                   np.sum([init_point,[0, size, 0]],axis = 0).tolist(),
                   np.sum([init_point,[size, size, size]],axis = 0).tolist()]
    # 1
    init = init_points[0]
    color = colors[0]
    # x, y 
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]+offset*j) 
            xyz.append(init[1]+offset*k) 
            xyz.append(init[2]) 
            p = Point(color, xyz)
            points.append(p) 
    # 2
    init = init_points[1]
    color = colors[1]
    # -y, z 
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]) 
            xyz.append(init[1]+offset*(-j)) 
            xyz.append(init[2]+offset*k) 

            p = Point(color, xyz)
            points.append(p) 
    # 3
    init = init_points[2]
    color = colors[2]
    # -x, -z 
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]+offset*(-j)) 
            xyz.append(init[1]) 
            xyz.append(init[2]+offset*(-k)) 
            p = Point(color, xyz)
            points.append(p) 
    # 4
    init = init_points[3]
    color = colors[3]
    # y, -z 
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]) 
            xyz.append(init[1]+offset*j) 
            xyz.append(init[2]+offset*(-k)) 
            p = Point(color, xyz)
            points.append(p) 
    # 5
    init = init_points[4]
    color = colors[4]
    # z, x
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]+offset*k) 
            xyz.append(init[1]) 
            xyz.append(init[2]+offset*j) 
            p = Point(color, xyz)
            points.append(p) 
    # 6
    init = init_points[5]
    color = colors[5]
    # -y, -x
    for j in range(points_num+1):
        for k in range(points_num+1):
            xyz = []
            xyz.append(init[0]+offset*(-k)) 
            xyz.append(init[1]+offset*(-j)) 
            xyz.append(init[2]) 
            p = Point(color, xyz)
            points.append(p) 
    return points