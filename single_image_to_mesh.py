import open3d
import numpy as np
import argparse
import cv2
import math
import os 

DEPTH_EXTENSION = '_depthcrop.png'
MASK_EXTENSION = '_maskcrop.png'
IMAGE_EXTENSION = '_crop.png'
USE_TXT = True

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

# create uv image using a RGB and a mesh
def create_uv_image(file_prefix,mesh):
    color = np.array(open3d.io.read_image(file_prefix + IMAGE_EXTENSION))
    im_height, im_width, _ = np.array(color).shape        
    uvmap = np.zeros((len(mesh.vertices), 2))
    for i, v in enumerate(mesh.vertices):
        v_unit = v #/ np.sqrt(sum(v**2))
        uvmap[i][0] = 0.5 + math.atan2(v_unit[2], v_unit[0]) / (2 * math.pi)
        uvmap[i][1] = 0.5 - math.asin(v_unit[1]) / math.pi

    texture = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for i, v in enumerate(mesh.vertices):
        u = int(uvmap[i][0] * 1024)
        v = int(uvmap[i][1] * 1024)
        x = int(v * im_width / 1024)
        y = int(u * im_height / 1024)
        color_for_pixel = color[y,x,:]
        texture[v][u] = color_for_pixel
    return texture

def main():

    parser=argparse.ArgumentParser(description="Create a mesh or a point cloud from a single image.")
    parser.add_argument("--image_location", default = '', help = 'Folder location')
    parser.add_argument("--image_prefix", required=True, help = 'Image Prefix - The part before _depth.png')
    parser.add_argument("--output", default = 'mesh', help = 'pointcloud, mesh or texture')
    parser.add_argument("--meshing_method", default = 'poisson', help = 'poisson, bpa or alpha')
    parser.add_argument("--is_cropped", default = 1, type = int, help = "Is the image is cropped? 1: True, 0: False")
    args=parser.parse_args()
    if args.image_location:
        os.chdir(args.image_location)

    if args.is_cropped == 0:
        global DEPTH_EXTENSION, MASK_EXTENSION, IMAGE_EXTENSION, USE_TXT
        DEPTH_EXTENSION = '_depth.png'
        MASK_EXTENSION = '_mask.png'
        IMAGE_EXTENSION = '.png'
        USE_TXT = False
    
    
    point_cloud = name_to_point_cloud(args.image_prefix)
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(point_cloud)
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
            final_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcd, 0.2)
        else:
            avg_nn_distance = np.mean(o3d_pcd.compute_nearest_neighbor_distance())
            r = avg_nn_distance * 3
            final_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3d_pcd,open3d.utility.DoubleVector([r, r * 2]))

        if args.output == 'texture':
            final_texture = create_uv_image(args.image_prefix,final_mesh)
            final_mesh.textures = [open3d.geometry.Image(final_texture)]
        vis.add_geometry(final_mesh)

    open3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()

if __name__ == "__main__":
    main()
