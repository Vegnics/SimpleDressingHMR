import argparse
import os
import sys
import json

import numpy as np
from ultralytics import YOLO
import open3d as o3d
from open3d.geometry import TriangleMesh,PointCloud
from open3d.visualization.rendering import Open3DScene
from typing import Tuple
import tensorflow as tf
import cv2

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import Config
from SMPL_estimator.model import Model
from SMPL_estimator.vis_util import preprocess_image,get_original
from smpl.compute_smpl_mesh import SMPL_mesh
from utils.drawing import draw_pose
from utils.scene_geometry import get_extrinsics
from utils.preprocessing import optimal_crop_from_kpts
from utils.postprocessing import EMA_filter


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





if __name__ == '__main__':
    print("FILENAME",__file__)
    """
    parser = argparse.ArgumentParser(description='Demo HMR2.0')

    parser.add_argument('--image', required=False, default='coco1.png')
    parser.add_argument('--model', required=False, default='base_model', help="model from logs folder")
    parser.add_argument('--setting', required=False, default='paired(joints)', help="setting of the model")
    parser.add_argument('--joint_type', required=False, default='cocoplus', help="<cocoplus|custom>")
    parser.add_argument('--init_toes', required=False, default=False, type=str2bool,
                        help="only set to True when joint_type=cocoplus")

    args = parser.parse_args()
    if args.init_toes:
        assert args.joint_type, "Only init toes when joint type is cocoplus!"
    """
    with open("./smpl/body_segmentation/removed_triangles.json","r") as file:
        removed_triangles = json.load(file)

    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath(os.path.join("SMPL_estimator","trained_models", "total_capture_model_paired"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "lsp"


    config = DemoConfig()
    # initialize model
    model = Model()
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00059.png")
    
    in_video_name = "videocap006.mp4"
    out_video_name = "test_0020.mp4"
    # Load the base mesh containing the uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")
    
    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    cam = cv2.VideoCapture(f"videos/{in_video_name}")
    
    videowriter = cv2.VideoWriter()
    retwrite = False

    pose_filter = EMA_filter(0.95,0.8,0.6,vtype="pose")
    beta_filter = EMA_filter(0.95) #0.9
    vertex_filter = EMA_filter(0.6,vtype="vertex") #0.7

    smpl_model = SMPL_mesh(model_type="neutral")

    while cam.isOpened():
        ret,img = cam.read()
        if ret:
            #img = cv2.imread(img_path)
            img = cv2.resize(img,(-1,-1),fx=0.7,fy=0.7,interpolation=cv2.INTER_LINEAR)   
            b_img = np.ascontiguousarray(img[:,:,::-1])
            back_img = o3d.geometry.Image(b_img)
            H,W = img.shape[0:2]
            #print(H,W) 
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            if retwrite==False:
                retwrite = videowriter.open(f"videos/{out_video_name}",fourcc,30.0,(int(2*W),H),isColor=True)
                if retwrite:
                    print("The video writer is opened")
            #retwriter = videowriter.open()
            ##module = tf.saved_model.load("./MoveNet_thunder")#
            ##img_rgb = np.copy(img[:,:,::-1])
            
            ##p0 = np.array([bbox[0][1],bbox[0][0]]).reshape(-1,2)
            ##person,retimg = movenet_detect(module,img_rgb[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:],input_size=256)
            ##kpts = person.kptsarray + p0

            results = posemodel(img,stream=False)
            kpts = np.float64(results[0].keypoints.data.numpy()[0,:,:])
            _sq_img = optimal_crop_from_kpts(img,kpts)
            
            original_img, input_img, params,_ = preprocess_image(_sq_img, config.ENCODER_INPUT_SHAPE[0])
            #original_img, input_img, params = preprocess_image(img[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:], config.ENCODER_INPUT_SHAPE[0])
            result = model.detect(input_img)
            cam_params = np.squeeze(result['cam'].numpy())[:3]
            betas = result['shape'].numpy()
            pose = result["pose_shape"].numpy()[0:-10]
            joints = np.squeeze(result["kp2d"].numpy())
            joints = ((joints + 1) * 0.5) * params['img_size']
            
            pose = pose_filter.filter(pose)
            betas = beta_filter.filter(betas)
            # Compute the vertices and triangles of the mesh
            #vertices = result['vertices'].numpy()
            vertices, triangles = smpl_model.get_mesh_from_params(pose,betas)
            vertices = vertex_filter.filter(vertices)


            # Create a Open3D Triangle Mesh
            #_,kpts= get_original(params,vertices,cam_params,joints)
            o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
            o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
            mesh = TriangleMesh(o3d_vertices,o3d_faces)
            mesh.triangle_uvs = base_mesh.triangle_uvs
            #text_img = cv2.imread("./texture_uv_data/textures/SMPLitex-texture-00000.png")#[:,:,::-1]
            
            #text_img = text_img.flip_vertical()
            mesh.textures = [text_img]
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
            geoms = [mesh]
            
            
            Kr = np.array([[2000.0, 0, 0.5*W],
                        [0, 2000.0, 0.5*H],
                        [0, 0, 1]])
            
            Kr2 = np.array([[3.03300909e+03, 0, (1.982444e+03)*0.62],
                        [0, 3.0365869e+03, (1.511572e+03)*0.62],
                        [0, 0, 1]])
            distcoeffs = np.array([3.27201245e-01,
                                   -2.78684166e+00,
                                   1.81260303e-03,
                                   -6.53144104e-03,
                                   8.0887056e+00])

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
        
            M = get_extrinsics(kpts,kpts3d,Kr)
            #print(M)
            
            mesh.vertices = o3d.utility.Vector3dVector(vertices +np.array([0.0,0.0,-0.5]).reshape(-1,3))
            mesh.compute_vertex_normals()
            mesh.remove_triangles_by_index(removed_triangles["triangles"])
      
            vis = o3d.visualization.rendering.OffscreenRenderer(width=W,height=H)
            IntrinsicsCam = o3d.camera.PinholeCameraIntrinsic(W,H,Kr)
            vis.setup_camera(IntrinsicsCam,M)

            material = o3d.visualization.rendering.MaterialRecord()
            material.base_metallic = 0.2  # Non-metallic
            material.base_roughness = 0.75  # High roughness for less gloss
            material.base_reflectance = 0.9  # Medium reflectance
            material.transmission = 0.3
            material.albedo_img = text_img
            material.shader = "defaultLit"
            #print(dir(material.shader))

            
            vis.scene.add_geometry("mesh",mesh,material)
            vis.scene.set_background([0.0,0.0,0.0,0.0],image=back_img)
            vis.scene.set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, (-0.4, 0.3, 0.6))#(0.577, -0.577, -0.577) (-0.4, -0.3, 0.4)
            

            O3dimg = vis.render_to_image()
            #print(np.asarray(O3dimg))
            rendered_img = np.asarray(O3dimg)[:,:,::-1]
            del vis
            
            
            #erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            #mask = cv2.inRange(rendered_img,(0,0,0),(10,10,10))
            #mask = cv2.erode(mask,erode_kernel,iterations=2)
            #masked = cv2.bitwise_and(rendered_img,rendered_img,mask=255-mask)+cv2.bitwise_and(img,img,mask=mask)
            
            fframe = np.hstack([rendered_img,draw_pose(img,kpts)])
            #print(fframe.shape)
            cv2.imshow("Frame",fframe)
            
            #cv2.imshow("Cropped",_sq_img)
            #mask = np.where(rendered_img[]==np.array([255,255,255],dtype=np.uint8),np.array([255,255,255],dtype=np.uint8),np.array([0,0,0],dtype=np.uint8))
            #cv2.imshow("Overlayed",masked)
            #cv2.imshow("With landmarks",draw_pose(img,kpts))
            #print("Frame")
            videowriter.write(fframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
cam.release()
videowriter.release()
cv2.destroyAllWindows()
    
    #img_out = draw_prediction_on_image(img, person, crop_reg)
    #o3d.visualization.draw_geometries(geoms)
    