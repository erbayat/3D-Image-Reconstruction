
Required Packages: open3d, numpy, argparse, opencv, os, glob

Required Files: Depth Images, RBG Images, loc.txt if images are cropped

Optional Files: Segmentation Masks

 --- single_image_to_mesh.py ---

It creates a point cloud and a mesh using single file. create_uv_image function does not give the expected result.


 --- image_list_to_mesh.py ---

It uses all images in a given folder to create a point cloud and a mesh. The output is not the expected 3D object. I will explain possible reasons on my presentation.
