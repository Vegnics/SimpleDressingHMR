import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
#from ld_data import smpl_vertices
import torch
from torchvision.transforms import Normalize

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator_SPIN.hmr import get_hmr
from SMPL_estimator_SPIN.imutils import crop
#from utils.renderer import Renderer
import SMPL_estimator_SPIN.config as config
import SMPL_estimator_SPIN.constants as constants
from smpl.compute_smpl_mesh import smpl_mesh_from_params
import open3d as o3d
from open3d.geometry import TriangleMesh,PointCloud
from open3d.visualization.rendering import Open3DScene
from typing import Tuple
from ultralytics import YOLO
import json

import tensorflow as tf
#from Movenet_core import _personLandmarkResults
#from Movenet_drawing import draw_prediction_on_image
import cv2

from utils.drawing import draw_pose
from utils.scene_geometry import get_extrinsics
from utils.preprocessing import optimal_crop_from_kpts
from utils.postprocessing import EMA_filter


FLT_EPSILON = sys.float_info.epsilon

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def squarify_img(img,kpnts):
    _kpnts = kpnts[:,0:2][kpnts[:,2]>0.6]
    _size_body = 210#140
    rem = (224-_size_body)/2
    H,W = img.shape[0:2]
    bbox_x0 = np.min(_kpnts[:,0]) 
    bbox_y0 = np.min(_kpnts[:,1])
    bbox_x1 = np.max(_kpnts[:,0])
    bbox_y1 = np.max(_kpnts[:,1])
    bbox_h = bbox_y1-bbox_y0
    bbox_w = bbox_x1-bbox_x0
    c_hip = 0.5*(kpnts[11,0:2]+kpnts[12,0:2])
    scale = max(bbox_h,bbox_w)/224
    if bbox_h>bbox_w:
        y_min = int(c_hip[1]-bbox_h/2-rem*scale)
        y_max = int(c_hip[1]+bbox_h/2+rem*scale)
        x_min = int(c_hip[0]-bbox_h/2-rem*scale)#int(max(bbox_x0-40,0))#
        x_max = int(c_hip[0]+bbox_h/2+rem*scale)#int(min(bbox_x1+40,W-1))#
        #left = 0
        #right = 0
    else:
        x_min = int(c_hip[0]-bbox_w/2-rem*scale)
        x_max = int(c_hip[0]+bbox_w/2+rem*scale)
        y_min = int(c_hip[1]-bbox_w/2-rem*scale)#int(bbox_y0)#
        y_max = int(c_hip[1]+bbox_w/2+rem*scale)#int(bbox_y1)#
        #top = 0
        #bottom = 0
    """
    N = max(H,W)
    top = (N-H)//2
    bottom = top
    left = (N-W)//2
    right = left
    """
    if y_min < 0:
        top = -y_min
        y_min = 0 
    else:
        top = 0
    if y_max > H-1:
        bottom = y_max-H
        y_max = H-1
    else:
        bottom = 0

    if x_min < 0:
        left = -x_min
        x_min = 0 
    else:
        left = 0
    if x_max > W-1:
        right = x_max-W
        x_max = W-1
    else:
        right = 0
    print("PADDING ",top,bottom,left,right)
    print("NBBOX:", x_min,x_max,y_min,y_max)
    #return cv2.resize(cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE),
    #                  (224,224),cv2.INTER_LINEAR)
    return np.copy(img[y_min:y_max,x_min:x_max,:])
    #return cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE)

def get_extrinsics(kpts,kpts3d,Kr,distcoeffs=np.array([0,0,0,0])):
    kpts2d = kpts[:,0:2][kpts[:,2]>0.6]
    kpts3d = kpts3d[kpts[:,2]>0.6]
    ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts2d,Kr,distcoeffs,flags=cv2.SOLVEPNP_SQPNP)
    rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts2d,Kr,distcoeffs,rvec,tvec,
                                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,100,10*FLT_EPSILON))
    _tvec = np.vstack([tvec,np.array([[1.0]])]).reshape(4,1) 
    _rmat = np.vstack([cv2.Rodrigues(rvec)[0],np.zeros((1,3))])
    return np.hstack([_rmat,_tvec])

def draw_pose(img,kpnts):
    _img = np.copy(img)
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (100,80,50),
    (0, 2): (50,80,100),
    (1, 3): (100,80,50),
    (2, 4): (50,80,100),
    (0, 5): (100,80,50),
    (0, 6): (50,80,100),
    (5, 7): (100,80,50),
    (7, 9): (100,80,50),
    (6, 8): (50,80,100),
    (8, 10): (50,80,100),
    (5, 6): (10,100,100),
    (5, 11): (100,80,50),
    (6, 12): (50,80,100),
    (11, 12): (10,100,100),
    (11, 13): (100,80,50),
    (13, 15): (100,80,50),
    (12, 14): (50,80,100),
    (14, 16): (50,80,100)
}
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        x0=int(kpnts[edge_pair[0],0])
        y0=int(kpnts[edge_pair[0],1])
        x1=int(kpnts[edge_pair[1],0])
        y1=int(kpnts[edge_pair[1],1])
        cv2.line(_img,(x0,y0),(x1,y1),color,3)
    for pnt in kpts:
        cv2.circle(_img,(int(pnt[0]),int(pnt[1])),7,(0,0,255),-1)
    return _img

def process_image(img_rgb, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = img_rgb.copy()

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

def movenet_detect(module, img_rgb, retPerson=True,input_size=192):
    """Runs landmark detection on an input image.

    Args:
      img_rgb: A [height, width, 3] input image containing only one person.
      module: A TensorFlow loaded model.
      retPerson: A boolean that can be set to True if the desired output is a _personLandmarkResults() object
      or can be set to False to return a Numpy ndarray.
      input_size: 192 if using lightning version OR 256 if using the thunder one

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint coordinates and scores OR a
      _personLandmarkResults() object according to the retPerson boolean.
    """
    # Convert the input image to a tensor with the desired size
    input_image = tf.expand_dims(img_rgb, axis=0)
    input_image = tf.image.resize(input_image, (input_size, input_size))
    model = module.signatures['serving_default']
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    _img = tf.cast(input_image[0,:,:,:],tf.uint8).numpy()
    if not retPerson:
        return keypoints_with_scores
    else:
        person = _personLandmarkResults()
        person.fromTensor(keypoints_with_scores, img_rgb.shape)
        return person,_img[:,:,::-1]




if __name__ == '__main__':
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

    with open("./smpl/body_segmentation/removed_triangles.json","r") as file:
        removed_triangles = json.load(file)

    #img_in = cv2.imread("./SMPL_estimator/images_test/slp_25.png")
    #img_in = cv2.imread("./SMPL_estimator/images_test/slp_06.png")
    #img_in = cv2.imread("./SMPL_estimator/images_test/slp_03.jpg")
    img_in = cv2.imread("./SMPL_estimator/images_test/slp_08.jpg")

    #bbox = [(66,75),(859,531)] # slp_25
    
    #bbox = [(125,160),(882,536)] # slp_06
    #bbox = [(101,102),(877,515)] # slp_03
    #bbox = [(120,134),(866,470)] # slp_08

    #module = tf.saved_model.load("./MoveNet_thunder")#
    img_rgb = np.copy(img_in[:,:,::-1])
    H,W = img_rgb.shape[0:2]
    
    #p0 = np.array([bbox[0][1],bbox[0][0]]).reshape(-1,2)
    #person,retimg = movenet_detect(module,img_rgb[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:],input_size=256)
    #kpts = person.kptsarray + p0

    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    results = posemodel(img_in,stream=False)
    kpts = np.float64(results[0].keypoints.data.numpy()[0,:,:])

    _sq_img = squarify_img(img_in,kpts)
    # initialize model
    chkpnt_path = "/home/amaranth/Desktop/NTU_2024/Computer_graphics/Final_project/simple_smpl/SMPL_estimator_SPIN/trained_models/model_checkpoint.pt"
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    device = torch.device('cpu')
    # Load pretrained model
    model = get_hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(chkpnt_path, map_location=torch.device('cpu'))
    #print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Preprocess input image and generate predictions
    img, norm_img = process_image(_sq_img, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        #pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        #pred_vertices = pred_output.vertices
    print("CAMERA")
    print("PredCamera",pred_camera.cpu().numpy())

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    print("CAMERA TRANSLATION",camera_translation)

    betas = pred_betas.numpy()
    pred_rotmat = pred_rotmat.numpy()
    rot_vecs = []
    for i in range(24):
        rot_vecs += list(np.reshape(cv2.Rodrigues(pred_rotmat[0,i,:,:])[0],(3,)))
    pose = np.array(rot_vecs)
    #print(camera_translation)
     # Compute the vertices and triangles of the mesh
    vertices, triangles = smpl_mesh_from_params(pose,betas)

    # Load the base mesh containing the uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")

    # Create a Open3D Triangle Mesh
    o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
    o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
    mesh = TriangleMesh(o3d_vertices,o3d_faces)
    #mesh.compute_vertex_normals()
    mesh.triangle_uvs = base_mesh.triangle_uvs
    #text_img = cv2.imread("./texture_uv_data/textures/SMPLitex-texture-00000.png")#[:,:,::-1]
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00000.png")
    #text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    geoms = [mesh]
    
    Kr = np.array([[2000.0, 0, 0.5*W],
                        [0, 2000.0, 0.5*H],
                        [0, 0, 1]])
 
    
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
    min_z = np.max(kpts3d[:,2])

    kpts3d += np.array([0.0,0.0,-0.9]).reshape(-1,3)
    print(kpts.shape,kpts.dtype,kpts3d.shape,kpts3d.dtype)
    
    #ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts,Kr,np.array([0,0,0,0]),flags=cv2.SOLVEPNP_SQPNP)
    
    #rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts,Kr,np.array([0,0,0,0]),rvec,tvec,
    #                                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,80,100*FLT_EPSILON))
    
    M = get_extrinsics(kpts,kpts3d,Kr)
    #if ret: print(M)

    img_pose = draw_pose(img_in,kpts)
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices +np.array([0.0,0.0,-0.9]).reshape(-1,3))
    mesh.compute_vertex_normals()
    mesh.remove_triangles_by_index(removed_triangles["triangles"])
    vis = o3d.visualization.rendering.OffscreenRenderer(width=576,height=1024)
    #vis = o3d.visualization.rendering.Visualizer()
    IntrinsicsCam = o3d.camera.PinholeCameraIntrinsic(576,1024,Kr)
    vis.setup_camera(IntrinsicsCam,M)

    material = o3d.visualization.rendering.MaterialRecord()
    material.base_metallic = 0.5  # Non-metallic
    material.base_roughness = 0.6  # High roughness for less gloss
    material.base_reflectance = 0.8  # Medium reflectance
    material.albedo_img = text_img
    material.shader = "defaultLit"
    

    vis.scene.add_geometry("mesh",mesh,material)
    vis.scene.set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, (-0.4, 0.3, 0.6))
    #center = mesh.get_center()  
    #camera = vis.scene.camera
    #camera.look_at(center, center + [0, 0, 1], [0, 1, 0])
    #camera.set_projection(60.0, 576 / 1024, 0.1, 10.0,o3d.visualization.rendering.Camera.FovType.Vertical)

    #vis.scene = scene
    

    O3dimg = vis.render_to_image()
    #print(np.asarray(O3dimg))
    rendered_img = np.asarray(O3dimg)[:,:,::-1]
    cv2.imshow("Rendered",rendered_img)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.inRange(rendered_img,(0,0,0),(230,230,230))
    mask = cv2.erode(mask,erode_kernel,iterations=2)
    masked = cv2.bitwise_and(rendered_img,rendered_img,mask=mask)+cv2.bitwise_and(img_in,img_in,mask=255-mask)
    cv2.imshow("With landmarks",draw_pose(masked,kpts))
    #mask = np.where(rendered_img[]==np.array([255,255,255],dtype=np.uint8),np.array([255,255,255],dtype=np.uint8),np.array([0,0,0],dtype=np.uint8))
    cv2.imshow("Overlayed",masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #img_out = draw_prediction_on_image(img, person, crop_reg)
    #o3d.visualization.draw_geometries(geoms)
    