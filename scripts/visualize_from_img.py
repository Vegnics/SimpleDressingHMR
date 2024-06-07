import argparse
import os
import sys

import numpy as np

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import HMRConfig
from SMPL_estimator.model import HMRModel
from utils.preprocessing import HMR_preprocess_image,optimal_crop_from_kpts
from smpl.compute_smpl_mesh import SMPL_mesh
import open3d as o3d
from open3d.geometry import TriangleMesh
from ultralytics import YOLO
from typing import Tuple
import tensorflow as tf
import cv2


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Visualizer_ImageToTexturedMesh Demo')
    parser.add_argument('--image', required=False, default='cheering_L.png')
    parser.add_argument('--model', required=False, default='male', help="male, female, or neutral SMPL models")
    parser.add_argument('--textimg', required=False, default='SMPLitex-texture-00000.png', help="Name of the texture image located at the texture_uv_data/textures folder")

    args = parser.parse_args()

    class DemoConfig(HMRConfig):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath(os.path.join("SMPL_estimator","trained_models", "total_capture_model"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "lsp"


    # Load the HMR model
    config = DemoConfig()
    hmr_model = HMRModel()

    # Load the input image
    img = cv2.imread(os.path.join("./images_test/",args.image))
    img_rgb = np.copy(img[:,:,::-1])
    
    # Load the SMPL model
    smpl_model = SMPL_mesh(model_type=args.model)

    # Human Pose estimation
    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    landmark_results = posemodel(img,stream=False)
    kpts = np.float64(landmark_results[0].keypoints.data.numpy()[0,:,:])

     # Crop the input frame according to the detected keypoints
    cropped_img = optimal_crop_from_kpts(img,kpts)

    # Pre-process the cropped image and apply HMR
    original_img, input_img, params = HMR_preprocess_image(cropped_img, config.ENCODER_INPUT_SHAPE[0])
    hmr_result = hmr_model.detect(input_img)
    betas = hmr_result['shape'].numpy()
    pose = hmr_result["pose_shape"].numpy()[0:-10]

    # Compute the vertices and triangles of the SMPL mesh
    vertices, triangles = smpl_model.get_mesh_from_params(pose,betas)

    # Load the base mesh containing the texture uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")

    # Create a Open3D Triangle Mesh
    o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
    o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
    mesh = TriangleMesh(o3d_vertices,o3d_faces)
   
    # Process the SMPL mesh, compute vertex normals, apply texture mapping
    mesh.compute_vertex_normals()
    mesh.triangle_uvs = base_mesh.triangle_uvs
    text_img = o3d.io.read_image(os.path.join("./texture_uv_data/textures/",args.textimg))
    text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    geoms = [mesh]
    
    o3d.visualization.draw_geometries(geoms)
    