from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2 as cv
from time import time
import natsort as ns

import cv3D as c3

def load_data():
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    return images_df, train_df, points3D_df, point_desc_df

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID") # [points_ID, avg_descriptors, XYZ, RGB] 
    return desc

if __name__ == "__main__":
    # load data
    images_df, train_df, points3D_df, point_desc_df = load_data() 
    images_df = images_df.reindex(index= ns.order_by_index(images_df.index, ns.index_natsorted(images_df["NAME"])) )
    images_df = images_df.reset_index(drop = True)

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df) # [points_ID, avg_descriptors, XYZ, RGB](in train) 
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    Trans_error = []
    Rot_error = []
    Translations = []
    Rotations = []
    # Load the total 130 validation images
    totalStartTime = time()
    for idx in range(163,293):
    # for idx in range(200,201):
        print(idx)
        startTime = time()

        # obtain the image ground truth
        ground_truth = images_df.loc[images_df.index==idx]
        image_id = ground_truth["IMAGE_ID"].values
        Rot_q_gt = ground_truth[["QX","QY","QZ","QW"]].values
        Trans_gt = ground_truth[["TX","TY","TZ"]].values

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==image_id[0]]
        kp_query = np.array(points["XY"].to_list()) # key points x,y in image coordinates
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32) # key points descriptors in image coordinates

        # Find correspondance and solve pnp
        cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
        distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

        pose = c3.poseEstimator(cameraMatrix, distCoeffs)
        points2Ds_undistort, points3Ds = pose.featureMatching((kp_query, desc_query),(kp_model, desc_model))
    
        Rot, Trans = pose.solveP3PRANSAC(points2Ds_undistort, points3Ds, iterations=None, threshold=0.1)  
        
        # estimated rotation matrix to axis angle
        
        Rot_vec = R.from_matrix(Rot).as_rotvec(degrees=True)
        # estimated translation (c to w) to (w to c) 
        
        Trans_wc = -np.dot(Rot,Trans)
        # groundtruth quaternion to axis angle
        # Rot_q_gt[:,[0,1,2,3]] = Rot_q_gt[:,[1,2,3,0]] # w,x,y,z -> x,y,z,w
        Rot_vec_gt = R.from_quat(Rot_q_gt).as_rotvec(degrees=True)
        # calculate and concatenate the error of translation & rotation
        trans_err = np.linalg.norm(Trans_wc-Trans_gt.reshape((3,1))) 
        # print(np.vdot(Rot_vec, Rot_vec_gt))
        # print(np.linalg.norm(Rot_vec))
        # print(np.linalg.norm(Rot_vec_gt))

        if np.vdot(Rot_vec, Rot_vec_gt) < 0:        
            rot_err = abs((360-np.linalg.norm(Rot_vec))-np.linalg.norm(Rot_vec_gt)) # kind of weird
        else:
            rot_err = abs(np.linalg.norm(Rot_vec)-np.linalg.norm(Rot_vec_gt)) # kind of weird
        
        Translations.append(Trans_wc)
        Rotations.append(Rot_vec)
        Trans_error.append(trans_err)         
        Rot_error.append(rot_err)
        
        endTime = time()
        print("Time elapsed : ",endTime-startTime)
    
    totalEndTime = time()
    print("Total time elapsed : ",totalEndTime-totalStartTime)
    
    Rot_error_med = np.median(np.array(Rot_error))
    Trans_error_med = np.median(np.array(Trans_error))
    print("Rotation median error : ",Rot_error_med)     
    print("Translation median error : ",Trans_error_med)     

    ##  plot the trajectory
    # list to np.array
    Translations = np.array(Translations)
    Rotations = np.array(Rotations)

    # save the camera translation and rotation
    np.save("npy/camera_Trans.npy", Translations)
    np.save("npy/camera_Rot.npy", Rotations)
    
