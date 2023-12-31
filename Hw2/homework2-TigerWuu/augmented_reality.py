import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2 as cv
import natsort as ns

#####################################
##### HW1:Q2-1: Augment Reality #####
#####################################
def load_data():
    points3D_df = pd.read_pickle("data/points3D.pkl")
    images_df = pd.read_pickle("data/images.pkl")
    images_df = images_df.reindex(index= ns.order_by_index(images_df.index, ns.index_natsorted(images_df["NAME"])) )
    images_df = images_df.reset_index(drop = True)
    return  images_df, points3D_df

def load_rot_trans_data():
    Translations = np.load("npy/camera_Trans.npy")
    Translations_gt = np.load("npy/camera_Trans_gt.npy")
    Rotations = np.load("npy/camera_Rot.npy")
    Rotations_gt = np.load("npy/camera_Rot_gt.npy")
    return Translations, Rotations, Translations_gt, Rotations_gt

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

if __name__=="__main__":
    # load data
    images_df, points3D_df = load_data()
    # camera_trajectory_gt(images_df)
    
    Translations, Rotations, Translations_gt, Rotations_gt = load_rot_trans_data()

    # define colors 
    dark_blue = [255, 70, 50]
    cyan = [240, 240, 90]
    magenta = [170, 70, 250]
    dark_green = [50, 140, 50]
    red = [0 , 0, 255]
    green = [0, 255, 0]
    colors = [cyan, green, red, dark_green, magenta,dark_blue ]
    # camera intrinsic 
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
   
    # generate cube points
    init_point = [1.45, -1.60, 1.5]
    side_points_num = 7
    size = 0.25
    # init_point2 = [3, -1.15, 2.6]
    # side_points_num2 = 5
    # size2 = 0.1

    fourcc = cv.VideoWriter_fourcc(*'xvid')
    ar_out = cv.VideoWriter('AR-cool.mp4', fourcc, 20.0, (1080, 1920))
    image_num = len(Translations)
    for i in range(image_num):
        cube_points = generate_cube_points(side_points_num, init_point, size, colors)
        Trans = Translations[i]
        Rot_matrix = R.from_rotvec(Rotations[i], degrees=True).as_matrix()

        for j in range(len(cube_points)):
            cube_points[j].xyz_camera = np.dot(Rot_matrix, cube_points[j].xyz_world)+Trans
            uv_homo_unnormal = np.dot(cameraMatrix, cube_points[j].xyz_camera)
            cube_points[j].uv_homo = uv_homo_unnormal/uv_homo_unnormal[2][0]

        # sort voxel by depth
        cube_points.sort(key = lambda x:x.xyz_camera[2][0], reverse=True)
       
        # paint voxel in the image
        print("images:", i)
        fname = ((images_df.loc[images_df.index == (i + 163)])["NAME"].values)[0]
        image = cv.imread("data/frames/"+fname)
        for k in range(len(cube_points)):
            u = int(cube_points[k].uv_homo[0][0])
            v = int(cube_points[k].uv_homo[1][0])
            image_ar = cv.circle(image, (u,v), 5, cube_points[k].color, -1)
        ar_out.write(image_ar)
        
        cv.namedWindow("AR cool", cv.WINDOW_NORMAL)
        cv.imshow("AR cool", image_ar)
        cv.waitKey(1)
    
    ar_out.release()
    cv.destroyAllWindows()



