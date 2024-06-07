import argparse
import os
import sys

import numpy as np
from ultralytics import YOLO

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import HMRConfig
from SMPL_estimator.model import HMRModel
from utils.scene_geometry import CameraModel
from utils.preprocessing import optimal_crop_from_kpts,HMR_preprocess_image
from utils.drawing import draw_pose
from smpl.compute_smpl_mesh import SMPL_mesh
import open3d as o3d
from open3d.geometry import TriangleMesh
from open3d.visualization.rendering import Open3DScene
from typing import Tuple
import tensorflow as tf
import json
import cv2

FLT_EPSILON = sys.float_info.epsilon

# SMPL mesh vertex indexes used to compute the 17 COCO keypoint locations
smpl_vertices=[[331],
          [6350,5333],
          [5361,5121],
          [5705,5560],
          [1815,3008],
          [1910,1647],
          [2242,2244],
          [6538,4973],
          [4523,4530],
          [6723,6603],
          [3091,915],
          [1050,1046],
          [3331,3323],
          [6260],
          [2800],
          [6887],
          [546]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageToTexturedMesh Demo')

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


    

    with open("./smpl/body_segmentation/removed_triangles.json","r") as file:
        removed_triangles = json.load(file)

    img_path = os.path.join("./images_test/",args.image)
    img = cv2.imread(img_path)   
    b_img = np.ascontiguousarray(img[:,:,::-1])
    back_img = o3d.geometry.Image(b_img)
    H,W = img.shape[0:2] 

    # Load the texture image
    text_img = o3d.io.read_image(os.path.join("./texture_uv_data/textures",args.textimg))

    # Camera model
    camera = CameraModel()

    # Load the SMPL model
    smpl_model = SMPL_mesh(model_type=args.model)

    # Human Pose estimation
    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    landmark_results = posemodel(img,stream=False)
    landmark_results = landmark_results[0].cpu()
    kpts = np.float64(landmark_results.keypoints.data.numpy()[0,:,:])

    # Crop the input frame according to the detected keypoints
    cropped_img = optimal_crop_from_kpts(img,kpts)
    
    # Initialize model for Human Mesh Recovery (i.e. estimation of the parameters for SMPL)
    config = DemoConfig()
    hmr_model = HMRModel()

    # Apply HMR to the cropped image
    original_img, input_img, params= HMR_preprocess_image(cropped_img, config.ENCODER_INPUT_SHAPE[0])
    hmr_results = hmr_model.detect(input_img)
    
    # Obtain the estimated SMPL pose and shape parameters 
    #cam_params = np.squeeze(result['cam'].numpy())[:3]
    betas = hmr_results['shape'].numpy()
    pose = hmr_results["pose_shape"].numpy()[0:-10]

    # Compute the vertices and triangles of the mesh using SMPL
    vertices, triangles = smpl_model.get_mesh_from_params(pose,betas)

    # Load the base mesh containing the texture uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")

    # Create a Open3D Triangle Mesh with the data provided by SMPL
    o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
    o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
    mesh = TriangleMesh(o3d_vertices,o3d_faces)
    mesh.triangle_uvs = base_mesh.triangle_uvs
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    
    Kr = np.array([[100000.0, 0, 0.5*W],
                    [0, 100000.0, 0.5*H],
                    [0, 0, 1]])
 
    # Compute the 3D locations of the 17 COCO keypoints from the 3D mesh vertices
    _kpts3d = []
    for k,vert in enumerate(smpl_vertices):
        if len(vert)>1:
            v0 = vertices[vert[0]]
            v1 = vertices[vert[1]]
            cnt = np.mean([v0,v1],axis=0)
        else:
            cnt = vertices[vert[0]]
        _kpts3d.append(cnt)
    orderk = [0,14,13,16,15,4,1,5,2,6,3,10,7,11,8,12,9]
    kpts3d = np.array([_kpts3d[k] for k in orderk])
    kpts3d += np.array([0.0,0.0,-0.5]).reshape(-1,3)
    
    # Compute camera extrinsics
    M = camera.get_extrinsics(kpts,kpts3d,(H,W))
    
    # Compute vertex normals and remove the triangle faces corresponding to the head,
    # neck, and feet 
    mesh.vertices = o3d.utility.Vector3dVector(vertices +np.array([0.0,0.0,-0.5]).reshape(-1,3))
    mesh.compute_vertex_normals()
    mesh.remove_triangles_by_index(removed_triangles["triangles"])
    
    # Initialize an Open3D renderer and set the camera model
    vis = o3d.visualization.rendering.OffscreenRenderer(width=W,height=H)
    IntrinsicsCam = o3d.camera.PinholeCameraIntrinsic(W,H,Kr)
    vis.setup_camera(IntrinsicsCam,M)

    # Set the material caracteristics of the 3D mesh, including the texture image
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_metallic = 0.5  # Non-metallic
    material.base_roughness = 0.6  # High roughness for less gloss
    material.base_reflectance = 0.8  # Medium reflectance
    material.albedo_img = text_img
    material.shader = "defaultLit"

    # Set the background image, and modify the orientation of the sun light
    vis.scene.add_geometry("mesh",mesh,material)
    vis.scene.set_background([0.0,0.0,0.0,1.0],image=back_img)
    vis.scene.set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, (-0.4, 0.3, 0.6))

    # Render the 3D mesh to an image
    O3dimg = vis.render_to_image()
    rendered_img = np.asarray(O3dimg)[:,:,::-1]

    cv2.imshow("Rendered",rendered_img)
    cv2.imshow("With landmarks",draw_pose(img,kpts))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    