import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2 as cv
import natsort as ns

#######################################
##### HW1:Q1-3: Camera Trajectory #####
#######################################
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

def camera_trajectory_gt(images_df):

    camera_trans_gt = []
    camera_Rot_gt = []
    
    for idx in range(163,293):
        ground_truth = images_df.loc[images_df.index==idx]
        Rot_q_gt = ground_truth[["QX","QY","QZ","QW"]].values
        Trans_gt = ground_truth[["TX","TY","TZ"]].values
        
        Rot_vec_gt = R.from_quat(Rot_q_gt).as_rotvec(degrees=True)

        camera_trans_gt.append(Trans_gt.reshape((3,1)))
        camera_Rot_gt.append(Rot_vec_gt[0])
    camera_trans_gt = np.array(camera_trans_gt)
    camera_Rot_gt = np.array(camera_Rot_gt)
    np.save("npy/camera_Trans_gt.npy", camera_trans_gt)
    np.save("npy/camera_Rot_gt.npy", camera_Rot_gt)

def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)    
    return pcd

def load_world_axes():
    worldaxes = o3d.geometry.LineSet()
    worldaxes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    worldaxes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    worldaxes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return worldaxes


def generate_trajetory(transPoints_wc, Rotation_vecs, colors):
    trajectory = o3d.geometry.LineSet()
    linewises = []
    linecolors = []
    transPoints_cw = []
    for i in range(len(transPoints_wc)-1):
        Rotation_matrix = R.from_rotvec(Rotation_vecs[i], degrees = True).as_matrix()
        transPoints_cw.append(np.dot(-np.linalg.inv(Rotation_matrix),transPoints_wc[i]))
        linewise = [i,i+1]
        linewises.append(linewise) 
        linecolors.append(colors)

    trajectory.points = o3d.utility.Vector3dVector(transPoints_cw)
    trajectory.lines  = o3d.utility.Vector2iVector(linewises)# X, Y, Z
    trajectory.colors = o3d.utility.Vector3dVector(linecolors) # R, G, B
    return trajectory

def generate_pyramid(transPoint_wc, Rotation_vec, colors, scale, ppo):
    pyramid = o3d.geometry.LineSet()
    edgecolors = []
    ox = ppo[0]*scale
    oy = ppo[1]*scale
    z = ppo[2]
    
    Rotation_matrix = R.from_rotvec(Rotation_vec, degrees = True).as_matrix()
    transPoint_cw = np.dot(-np.linalg.inv(Rotation_matrix),transPoint_wc)

    pyramid_corners_c = np.array([[0  ,   0, 0],
                                  [ox ,  oy, z],
                                  [ox , -oy, z],
                                  [-ox, -oy, z],
                                  [-ox,  oy, z]]) 

    # rotate the pyramid
    for i in range(0,5):
        pyramid_corners_c[i] = (np.dot(np.linalg.inv(Rotation_matrix), pyramid_corners_c[i].reshape((3,1))) + transPoint_cw).reshape((1,3)) 
    
    pyramid_edges = [[0,1],[0,2],[0,3],[0,4],[4,1],[1,2],[2,3],[3,4]] 
    for _ in range(8):
        edgecolors.append(colors)
    
    pyramid.points = o3d.utility.Vector3dVector(pyramid_corners_c)
    pyramid.lines  = o3d.utility.Vector2iVector(pyramid_edges)# X, Y, Z
    pyramid.colors = o3d.utility.Vector3dVector(edgecolors) # R, G, B
    return pyramid

def generate_image(transPoint_wc, Rotation_vec, image, ppo, resize_scale, scale):
    image = cv.resize(image, (int(image.shape[1]*resize_scale), int(image.shape[0]*resize_scale)), interpolation=cv.INTER_AREA)
    rgb = []
    xyz = []
    u = image.shape[1] # 1080* 0.1 
    v = image.shape[0] # 1920* 0.1
    z = ppo[2] 
    # print("u : ", u) 
    # print("v : ", v) 
    for i in range(u):
        for j in range(v):
            rgb.append(image[j][i].reshape((3,)))
            xyz.append([(i-u/2)* scale/resize_scale,(j-v/2)* scale/resize_scale, z]) # ??
    rgb = np.array(rgb)/255
    xyz = np.array(xyz)
    # print(rgb)
    # print(xyz)
    Rotation_matrix = R.from_rotvec(Rotation_vec, degrees = True).as_matrix()
    transPoint_cw = np.dot(-np.linalg.inv(Rotation_matrix),transPoint_wc)
    for i in range(len(xyz)):
        xyz[i] = (np.dot(np.linalg.inv(Rotation_matrix), xyz[i].reshape((3,1))) + transPoint_cw).reshape((1,3)) 
    
    pcd_image = o3d.geometry.PointCloud()
    pcd_image.points = o3d.utility.Vector3dVector(xyz)
    pcd_image.colors = o3d.utility.Vector3dVector(rgb)    
    return pcd_image

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

if __name__=="__main__":
    # load data
    images_df, points3D_df = load_data()
    camera_trajectory_gt(images_df)

    Translations, Rotations, Translations_gt, Rotations_gt = load_rot_trans_data()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # load axes
    # world_axes = load_world_axes()
    # vis.add_geometry(world_axes)
    # load NTU gate pointcloud
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    # define colors 
    dark_blue = [50/255, 70/255, 1]
    cyan = [90/255, 240/255, 240/255]
    magenta = [250/255, 70/255, 170/255]
    orange = [1, 200/255, 0]

    # plot the camera trajectory
    trajectories = generate_trajetory(Translations, Rotations, dark_blue)
    vis.add_geometry(trajectories)
    
    # plot the camera trajectory groundtruth
    # trajectories_gt = generate_trajetory(Translations_gt, Rotations_gt, magenta)
    # vis.add_geometry(trajectories_gt)

    # plot the quadrangular pyramid
    image_num = len(Translations)
    O = [540, 960, 0.5] # ox, oy, focal length(96 dpi)
    fov_x = 1 * np.pi/2 # assume hfov = 90 degree
    pyramid_scale = np.tan(fov_x/2)*0.5/O[0]
    # pyramid_scale = 0.0001 

    for i in range(image_num):
        print("images:", i)
        pyramid = generate_pyramid(Translations[i], Rotations[i], orange, scale=pyramid_scale , ppo=O) # translation : Nx3x1. ppo: principle point : [ox,oy, focal_length] 
        vis.add_geometry(pyramid)
        
        fname = ((images_df.loc[images_df.index == (i + 163)])["NAME"].values)[0]
        image = cv.imread("data/frames/"+fname)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pcd_image = generate_image(Translations[i], Rotations[i], image, ppo=O, resize_scale=0.1, scale=pyramid_scale) # translation : Nx3x1
        vis.add_geometry(pcd_image)
        
        # pyramid_gt = generate_pyramid(Translations_gt[i], Rotations_gt[i], cyan) # translation : Nx3x1
        # vis.add_geometry(pyramid_gt)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
 
    vis.run()
