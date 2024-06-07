import argparse
import os
import sys
import json

import numpy as np
from ultralytics import YOLO
import open3d as o3d
from open3d.geometry import TriangleMesh
from open3d.visualization.rendering import Open3DScene
from typing import Tuple
import tensorflow as tf
import cv2

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import HMRConfig
from SMPL_estimator.model import HMRModel
from smpl.compute_smpl_mesh import SMPL_mesh
from utils.drawing import draw_pose
from utils.scene_geometry import CameraModel
from utils.preprocessing import optimal_crop_from_kpts,HMR_preprocess_image
from utils.postprocessing import EMA_filter

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
    parser = argparse.ArgumentParser(description='VideoToTexturedMesh Demo')

    parser.add_argument('--model', required=False, default='male', help="male, female, or neutral SMPL models")
    parser.add_argument('--ivideo', required=True, default='null.mp4', help="Name of the input video located at the videos folder")
    parser.add_argument('--ovideo', required=True, default='null.mp4', help="Name of the input video located at the videos folder")
    parser.add_argument('--textimg', required=False, default='SMPLitex-texture-00000.png', help="Name of the texture image located at the texture_uv_data/textures folder")

    args = parser.parse_args()

    # Load the indices of the triangles to be removed from the the computed triangle mesh
    with open("./smpl/body_segmentation/removed_triangles.json","r") as file:
        removed_triangles = json.load(file)

    class DemoConfig(HMRConfig):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath(os.path.join("SMPL_estimator","trained_models", "total_capture_model_paired"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "lsp"

    # initialize model for Human Mesh Recovery (i.e. estimation of the parameters for SMPL)
    config = DemoConfig()
    hmr_model = HMRModel()

    # Load the texture image
    text_img = o3d.io.read_image(os.path.join("./texture_uv_data/textures",args.textimg))
    
    # Set the names of the input / output videos from the argparser
    in_video_name = args.ivideo #"videocap001.mp4"
    out_video_name = args.ovideo #"test_0039.mp4"

    # Load the base mesh containing the texture UV maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")
    
    # Human pose estimation models (used to obtain 2d landmarks)
    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    
    # Camera model
    camera = CameraModel()

    # Initialize a OpenCV videowriter
    videowriter = cv2.VideoWriter()
    retwrite = False

    # SMPL parameter filters
    pose_filter = EMA_filter(0.8,0.7,0.7,vtype="pose")
    beta_filter = EMA_filter(0.85,vtype="shape")

    # Load the SMPL model
    smpl_model = SMPL_mesh(model_type=args.model)

    # Start processing the input video
    videostream = cv2.VideoCapture(f"videos/{in_video_name}")
    while videostream.isOpened():
        ret,img = videostream.read()
        if ret:
            # Read a frame from the video stream, and generate an Open3D background image
            img = cv2.resize(img,(-1,-1),fx=0.85,fy=0.85,interpolation=cv2.INTER_LINEAR)   
            b_img = np.ascontiguousarray(img[:,:,::-1])
            back_img = o3d.geometry.Image(b_img)
            H,W = img.shape[0:2]
 
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            if retwrite==False:
                retwrite = videowriter.open(f"videos/{out_video_name}",fourcc,30.0,(int(2*W),H),isColor=True)
                if retwrite:
                    print("The video writer is opened!")

            # Human Pose estimation
            landmark_results = posemodel(img,stream=False)
            kpts = np.float64(landmark_results[0].keypoints.data.numpy()[0,:,:])
            
            # Crop the input frame according to the detected keypoints
            crop_img = optimal_crop_from_kpts(img,kpts)
            
            # Apply HMR to the cropped image
            original_img, input_img, params = HMR_preprocess_image(crop_img, config.ENCODER_INPUT_SHAPE[0])
            hmr_results = hmr_model.detect(input_img)
            
            # Obtain the estimated SMPL pose and shape parameters 
            betas = hmr_results['shape'].numpy()
            pose = hmr_results["pose_shape"].numpy()[0:-10]
            
            # Apply EMA filtering to the SMPL parameters
            pose = pose_filter.filter(pose)
            betas = beta_filter.filter(betas)
            
            # Compute the vertices and triangles of the 3D mesh using SMPL 
            vertices, triangles = smpl_model.get_mesh_from_params(pose,betas)

            # Create an Open3D Triangle Mesh with the data provided by SMPL
            o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
            o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
            mesh = TriangleMesh(o3d_vertices,o3d_faces)
            mesh.triangle_uvs = base_mesh.triangle_uvs
            mesh.textures = [text_img]
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))

            # Approximation of an Orthographic projection using pinhole camera parameters
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
            kpts3d += np.array([0.0,0.0,-0.6]).reshape(-1,3)

            # Compute camera extrinsics
            M = camera.get_extrinsics(kpts,kpts3d,(H,W))
            
            # Compute vertex normals and remove the triangle faces corresponding to the head,
            # neck, and feet 
            mesh.vertices = o3d.utility.Vector3dVector(vertices +np.array([0.0,0.0,-0.6]).reshape(-1,3))
            mesh.compute_vertex_normals()
            mesh.remove_triangles_by_index(removed_triangles["triangles"])
            mesh.remove_unreferenced_vertices()
            
            # Initialize an Open3D renderer and set the camera model
            vis = o3d.visualization.rendering.OffscreenRenderer(width=W,height=H)
            IntrinsicsCam = o3d.camera.PinholeCameraIntrinsic(W,H,Kr)
            vis.setup_camera(IntrinsicsCam,M)
            
            # Set the material caracteristics of the 3D mesh, including the texture image
            material = o3d.visualization.rendering.MaterialRecord()
            material.base_metallic = 0.2  # Non-metallic
            material.base_roughness = 0.8  # High roughness for less gloss
            material.base_reflectance = 0.78  # Medium reflectance
            material.base_anisotropy = 0.53 # 
            material.albedo_img = text_img # texture image
            material.shader = "defaultLit"

            # Set the background image, and modify the orientation of the sun light
            vis.scene.add_geometry("mesh",mesh,material)
            vis.scene.set_background([0.0,0.0,0.0,1.0],image=back_img)
            vis.scene.set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, (-0.4, 0.3, 0.6))# Default: (0.577, -0.577, -0.577)
            
            # Render the 3D mesh to an image
            O3dimg = vis.render_to_image()
            rendered_img = np.asarray(O3dimg)[:,:,::-1]
            del vis
            
            # Stack the rendered image with an image depicting the detected 2D landmarks 
            fframe = np.hstack([rendered_img,draw_pose(img,kpts)])
            cv2.imshow("Frame",fframe)
            videowriter.write(fframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    videostream.release()
    videowriter.release()
    cv2.destroyAllWindows()