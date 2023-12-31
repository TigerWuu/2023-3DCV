import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
        self.keep_running = True
        print("Intrisic : ",self.K)
        print("Distortion : ",self.dist)
        # print(os.path.join(args.input, '*.png'))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        T_last = np.array([[0,0,0]])
        magenta = [250/255, 70/255, 170/255]
        cyan = [90/255, 240/255, 240/255]
        dark_blue = [50/255, 70/255, 1]
        
        worldaxis = self.__load_world_axes()
        vis.add_geometry(worldaxis)
        O = [self.K[0][2], self.K[1][2], 0.5]
        fov_x = 1 * np.pi/2 # assume hfov = 90 degree
        pyramid_scale = np.tan(fov_x/2)*0.5/O[0]       
        
        while self.keep_running:
            try:
                R, T, img_k1= queue.get(block=False) # if true, stop getting r,t when queue is empty
                img_k1 = cv.cvtColor(img_k1, cv.COLOR_BGR2RGB)
                if R is not None:
                    # insert new camera pose here using vis.add_geometry()
                    T_vis = np.concatenate((T_last.reshape((1,3)), T.reshape((1,3))), axis=0)
                    trajectory = self.__generate_trajetory(T_vis, magenta)
                    pyramid = self.__generate_pyramid(T, R, dark_blue, scale=pyramid_scale, ppo=O)
                    pcd_image = self.__generate_image(T, R, img_k1 ,scale=pyramid_scale, ppo=O, resize_scale= 0.2)
                    vis.add_geometry(trajectory) 
                    vis.add_geometry(pyramid) 
                    vis.add_geometry(pcd_image) 
                    T_last = T
            except Exception as e:
                # print("error : ", e)
                pass
            
            self.keep_running = self.keep_running and vis.poll_events()
            # vis.update_renderer()
        vis.destroy_window()
        p.join()
   
    def process_frames(self, queue):
        R = np.eye(3, dtype=np.float64)
        T = np.zeros((3, 1), dtype=np.float64)
        t_last_norm = 1
        last_good_matches = []
        last_mask = []
        false_count = 0
        scale_to_1 = 1

        img_k = cv.imread(self.frame_paths[0])
        orb = cv.ORB_create() 
        kp1, des1 = orb.detectAndCompute(img_k, None)
        for frame_path in self.frame_paths[1:]:
            img_k1 = cv.imread(frame_path)
            # feature matching with orb
            kp2, des2 = orb.detectAndCompute(img_k1, None)
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # brutal force matching
            good_matches = matcher.match(des1, des2)
            good_matches = sorted(good_matches, key = lambda x:x.distance)
            # good_matches = good_matches[0:int(len(good_matches)/2)]
            # good_matches = []

            pointsk = np.array([kp1[m.queryIdx].pt for m in good_matches]) # the keypoint of the first image (query)
            pointsk_1 = np.array([kp2[m.trainIdx].pt for m in good_matches]) # the keypoint of the second image (train)
            kp_match = np.array([kp2[m.trainIdx] for m in good_matches]) # the keypoint of the second image (train)
            

            # estimate the essential matrix
            # undistort the 2D points         
            for i in range(len(pointsk)): 
                pointsk[i] = cv.undistortPoints(pointsk[i], self.K, self.dist, P=self.K)
                pointsk_1[i] = cv.undistortPoints(pointsk_1[i], self.K, self.dist, P=self.K)

            E, mask_e = cv.findEssentialMat(pointsk, pointsk_1, self.K)
            # decompose the essential matrix to get the R and T
            _, r, t, mask_rt, triPoints = cv.recoverPose(E, pointsk, pointsk_1, self.K, distanceThresh = 50)
            # print("mask :", mask_rt.shape)
            # print("3D :", triPoints.shape)
            mask = []
            for i in range(len(mask_e)):
                if mask_e[i][0] == 1 and mask_rt[i][0]==255:
                    mask.append(255)
                else:
                    mask.append(0)
            
            X_k_k1_Idx = []
            X_k1_k_Idx = []
            for i, m in enumerate(last_good_matches):
                if last_mask[i] == 255: # determine whethere it is inliers. 255 : inlier; 0 : outlier
                    for j, n in enumerate(good_matches):
                        if (np.array(kp1[m.trainIdx].pt) == np.array(kp1[n.queryIdx].pt)).all():
                            if mask[j] == 255:
                                X_k1_k_Idx.append(i)
                                X_k_k1_Idx.append(j)
                                break
                    # if len(X_k_k1_Idx) == 2:
                    #     print(X_k_k1_Idx)
                    #     print(X_k1_k_Idx)
                    #     break
                else:
                    continue
            points3D_num = len(X_k1_k_Idx) 
            avg_scale = 0
            count = 0
            if points3D_num >= 2:
                for i in range(len(X_k1_k_Idx)-1):
                    X_k1_k = last_triPoints[:,X_k1_k_Idx[i]]/last_triPoints[:,X_k1_k_Idx[i]][3]
                    X_p_k1_k = last_triPoints[:,X_k1_k_Idx[i+1]]/last_triPoints[:,X_k1_k_Idx[i+1]][3]               
                    X_k_k1 = np.dot(np.concatenate((last_r, last_t),axis=1),((triPoints[:,X_k_k1_Idx[i]]/triPoints[:,X_k_k1_Idx[i]][3]).reshape((4,1))))
                    X_p_k_k1 = np.dot(np.concatenate((last_r, last_t),axis=1),((triPoints[:,X_k_k1_Idx[i+1]]/triPoints[:,X_k_k1_Idx[i+1]][3]).reshape((4,1))))
                    scale = np.linalg.norm(X_k_k1-X_p_k_k1)/np.linalg.norm(X_k1_k-X_p_k1_k)*t_last_norm
                    if scale < 3 and scale > 0.3:
                        avg_scale +=scale # average the scale of inliers 
                        count+=1
                        # break
                    
                if avg_scale == 0:
                    avg_scale = 1
                    print("No proper scale!")
                else:
                    avg_scale /= count
                # scale /= int(len(X_k1_k_Idx)/2)
                # print(X_k1_k)
                # print(X_p_k1_k)
                # print(X_k_k1)
                # print(X_p_k_k1)
            else:
                print("Oops! : ", len(X_k1_k_Idx))
                false_count += 1
                avg_scale = 1
            
            # recover return r,t : performs a change of basis from the k camera's coordinate system to k+1 camera's coordinate system 
            # r : c1->c0 ? t : c0->c1 ? 
            r = np.linalg.inv(r)
            t = -np.dot(r,t)*avg_scale
            
            scale_to_1 *= avg_scale
            print("scale : ",avg_scale)
            print("scale to initial frame : ",scale_to_1)
            # calculate the pose relate to the first frame
            T = T+np.dot(R,t)
            R = np.dot(R, r)
            
            queue.put((R, T, img_k1))
            for i in range(len(kp_match)):
                img_k1 = cv.drawKeypoints(img_k1, [kp_match[i]], None, color=(128,0,255-i), flags=0)
            cv.imshow('frame', img_k1)
            if cv.waitKey(30) == 27: 
                self.keep_running = False
                print(self.keep_running)
                break

            kp1 = kp2
            des1 = des2

            t_last_norm = np.linalg.norm(t) 
            last_good_matches = good_matches
            last_triPoints = triPoints
            last_mask = mask 
            last_r = r
            last_t = t
            
        cv.destroyAllWindows()
        print("False : ", false_count)
    
    def __load_world_axes(self):
        worldaxes = o3d.geometry.LineSet()
        worldaxes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        worldaxes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
        worldaxes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
        return worldaxes
    
    def __generate_trajetory(self, transPoints_k_k1, linecolors):
        trajectory = o3d.geometry.LineSet()
        linewises = [[0,1]] 
        linecolors = [linecolors]
    
        trajectory.points = o3d.utility.Vector3dVector(transPoints_k_k1)
        trajectory.lines  = o3d.utility.Vector2iVector(linewises)# X, Y, Z
        trajectory.colors = o3d.utility.Vector3dVector(linecolors) # R, G, B
        return trajectory
    
    def __generate_pyramid(self, transPoint_k_k1, Rotation, colors, scale, ppo):
        pyramid = o3d.geometry.LineSet()
        edgecolors = []
        ox = ppo[0]*scale
        oy = ppo[1]*scale
        z = ppo[2]

        pyramid_corners_c = np.array([[0  ,   0, 0],
                                      [ox ,  oy, z],
                                      [ox , -oy, z],
                                      [-ox, -oy, z],
                                      [-ox,  oy, z]]) 

        # rotate the pyramid
        for i in range(0,5):
            pyramid_corners_c[i] = (np.dot(Rotation, pyramid_corners_c[i].reshape((3,1))) + transPoint_k_k1).reshape((1,3)) 
    
        pyramid_edges = [[0,1],[0,2],[0,3],[0,4],[4,1],[1,2],[2,3],[3,4]] 
        for _ in range(len(pyramid_edges)):
            edgecolors.append(colors)
    
        pyramid.points = o3d.utility.Vector3dVector(pyramid_corners_c)
        pyramid.lines  = o3d.utility.Vector2iVector(pyramid_edges)# X, Y, Z
        pyramid.colors = o3d.utility.Vector3dVector(edgecolors) # R, G, B
        return pyramid

    def __generate_image(self, transPoint_k_k1, Rotation, image, scale, ppo, resize_scale):
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
        for i in range(len(xyz)):
            xyz[i] = (np.dot(Rotation, xyz[i].reshape((3,1))) + transPoint_k_k1).reshape((1,3)) 
    
        pcd_image = o3d.geometry.PointCloud()
        pcd_image.points = o3d.utility.Vector3dVector(xyz)
        pcd_image.colors = o3d.utility.Vector3dVector(rgb)    
        return pcd_image
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="frames", help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='npy/camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
