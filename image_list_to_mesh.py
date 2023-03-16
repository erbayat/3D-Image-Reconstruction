import open3d
import numpy as np
import argparse
import cv2
import os 
import glob

DEPTH_EXTENSION = '_depthcrop.png'
MASK_EXTENSION = '_maskcrop.png'
IMAGE_EXTENSION = '_crop.png'
USE_TXT = True

#detector = cv2.ORB_create(scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=100, patchSize=31)
matcher = cv2.BFMatcher()
detector = cv2.xfeatures2d.SIFT_create()

# Form point cloud using single depth image.
def name_to_point_cloud(file_prefix):
    center = [319.5, 239.5]
    depth_matrix = cv2.imread(file_prefix + DEPTH_EXTENSION , cv2.IMREAD_UNCHANGED)
    [im_height, im_width] = depth_matrix.shape
    depth_matrix = depth_matrix.astype(float)
    constant = 570.3
    MM_PER_M = 1000
    if os.path.isfile(file_prefix + MASK_EXTENSION):
        mask_matrix = cv2.imread(file_prefix + MASK_EXTENSION, cv2.IMREAD_UNCHANGED)
        mask_matrix = (mask_matrix/np.amax(mask_matrix)).astype(int)
        depth_matrix = np.multiply(depth_matrix, mask_matrix)
    if os.path.isfile(file_prefix + '_loc.txt') and USE_TXT:
        with open(file_prefix + '_loc.txt', "r") as filestream:
            for line in filestream:
                loc = np.array(line.strip().split(',')).astype(int)
    else:
        loc = [1,1]
    point_cloud = np.zeros((im_height,im_width,3))
    xgrid = np.tile(np.arange(0,im_width),(im_height,1)) + (loc[0]-1) - center[0]
    ygrid = np.tile(np.reshape(np.arange(0,im_height),(im_height,1)),(1,im_width)) + (loc[1]-1) - center[1]
    point_cloud[:,:,0] = np.multiply(xgrid,depth_matrix)/constant/MM_PER_M
    point_cloud[:,:,1] = np.multiply(ygrid,depth_matrix)/constant/MM_PER_M
    point_cloud[:,:,2] = depth_matrix/MM_PER_M
    point_cloud = point_cloud.reshape(im_height*im_width,3)
    point_cloud = point_cloud[point_cloud[:,2] > 0]
    return point_cloud

# form combined point cloud using all images. 
def combine_point_clouds(image_prefix_list):
    K = np.array([[570.3, 0, 319.5], [0, 570.3, 239.5], [0, 0, 1]])
    R_combined = np.eye(3)
    t_combined = np.array([0, 0, 0]).reshape((3,1))
    combined_point_cloud = name_to_point_cloud(image_prefix_list[0])
    distance_between_images = 1
    for i in range(1,len(image_prefix_list)): # find correspondences to calculate essential matrix
        reference_image = image_prefix_list[i-distance_between_images]
        target_image = image_prefix_list[i]
        ref_img_color = cv2.imread(reference_image + IMAGE_EXTENSION)
        target_img_color = cv2.imread(target_image + IMAGE_EXTENSION)
        kp1, des1 = detector.detectAndCompute(ref_img_color, None)
        kp2, des2 = detector.detectAndCompute(target_img_color, None)
        matches = matcher.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), k=2)
        pts1 = []
        pts2 = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                if (m.trainIdx < len(kp1) and n.trainIdx < len(kp2)):
                    pts1.append(kp1[m.trainIdx].pt)
                    pts2.append(kp2[n.trainIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        if len(pts1) < 8:
            distance_between_images += 1 
            print('Fundamental matrix for', reference_image, target_image, 'could not be calculated.')
            continue
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if not type(F) is np.ndarray:
            distance_between_images += 1 
            print('Fundamental matrix for', reference_image, target_image, 'could not be calculated.')
            continue

        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if not (type(F) is np.ndarray and F.shape == (3,3)):
            distance_between_images += 1 
            print('Fundamental matrix for', reference_image, target_image, 'could not be calculated.')
            continue
        
        E = K.T @ F @ K
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K) # find rotation and translation
        R_combined = np.dot(R_combined, R) #R_combined and t_combined stands for R and t when the reference image is the first image.
        t_combined = np.dot(R_combined, t) + t_combined
        point_cloud_target = name_to_point_cloud(target_image)
        point_cloud_target = np.dot(R_combined, point_cloud_target.T).T + t_combined.T
        combined_point_cloud = np.concatenate((combined_point_cloud, point_cloud_target), axis=0)
    return combined_point_cloud

def main():

    parser=argparse.ArgumentParser(description="Create a mesh or a point cloud from a single image.")
    parser.add_argument("--image_location", default = '', help = 'Folder location')
    parser.add_argument("--output", default = 'mesh', help = 'pointcloud, mesh or texture')
    parser.add_argument("--image_count_limit", default=1000, type = int, help = "Maximum number of image used to create a point cloud")
    parser.add_argument("--is_cropped", default = 1, type = int, help = "Is the image is cropped? 1: True, 0: False")
    parser.add_argument("--meshing_method", default = 'poisson', help = 'poisson, bpa or alpha')
    
    args=parser.parse_args()
    if args.image_location:
        os.chdir(args.image_location)

    if args.is_cropped == 0:
        global DEPTH_EXTENSION, MASK_EXTENSION, IMAGE_EXTENSION, USE_TXT
        DEPTH_EXTENSION = '_depth.png'
        MASK_EXTENSION = '_mask.png'
        IMAGE_EXTENSION = '.png'
        USE_TXT = False

    image_prefix_list = []
    for file in glob.glob("*.txt"):
        image_prefix_list.append(file.replace('_loc.txt',''))
        if len(image_prefix_list) >= args.image_count_limit:
            break
        
    combined_pc = combine_point_clouds(image_prefix_list)
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(combined_pc)
    o3d_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    if args.output == 'pointcloud':
        vis.add_geometry(o3d_pcd)
    else: # from point clouds to a meshe
        if args.meshing_method == 'poisson':
            poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
            bbox = o3d_pcd.get_axis_aligned_bounding_box()
            final_mesh = poisson_mesh.crop(bbox)
        elif args.meshing_method == 'alpha':
            final_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcd, 0.1)
        else:
            avg_nn_distance = np.mean(o3d_pcd.compute_nearest_neighbor_distance())
            r = avg_nn_distance * 3
            final_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3d_pcd,open3d.utility.DoubleVector([r, r * 2]))
        vis.add_geometry(final_mesh)


    open3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()

if __name__ == "__main__":
    main()
