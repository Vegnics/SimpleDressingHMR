import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from ld_data import smpl_vertices
from ultralytics import YOLO

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import Config
from SMPL_estimator.model import Model
from SMPL_estimator.vis_util import preprocess_image,get_original
from smpl.compute_smpl_mesh import smpl_mesh_from_params
import open3d as o3d
from open3d.geometry import TriangleMesh,PointCloud
from typing import Tuple
import tensorflow as tf
from Movenet_core import _personLandmarkResults
from Movenet_drawing import draw_prediction_on_image
import cv2

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

def create_sphere(center:np.ndarray,radius:float = 0.05):
    sphere = TriangleMesh.create_sphere(radius,50)
    _sphere_verts = np.asarray(sphere.vertices)+np.reshape(center,(1,3))
    sphere.vertices = o3d.utility.Vector3dVector(_sphere_verts)
    sphere.paint_uniform_color(np.array([1.0,0.0,0.0]).reshape(3,1))
    sphere.compute_vertex_normals()
    return sphere

def find_closest_vertices(vertex_id,vertices,radius:float = 0.05):
    idxs = np.arange(vertices.shape[0])
    dists = np.linalg.norm(vertices-np.reshape(vertices[vertex_id],(1,3)),axis=-1)
    close_verts = idxs[dists<radius]
    mean_pos = np.mean(vertices[close_verts],axis=0)
    dists = np.linalg.norm(vertices-np.reshape(mean_pos,(1,3)),axis=-1)
    close_verts = idxs[dists<radius*0.75]
    _zip = list(zip(close_verts,dists))
    verts = sorted(_zip,key=lambda x:x[1])
    return [verts[i][0] for i in range(2)]

def squarify_img(img,kpnts):
    _kpnts = kpnts[:,0:2][kpnts[:,2]>0.6]
    _size_body = 130
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

def get_extrinsics(kpts,kpts3d,Kr):
    kpts2d = kpts[:,0:2][kpts[:,2]>0.6]
    kpts3d = kpts3d[kpts[:,2]>0.6]
    ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts2d,Kr,np.array([0,0,0,0]),flags=cv2.SOLVEPNP_SQPNP)
    rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts2d,Kr,np.array([0,0,0,0]),rvec,tvec,
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

def pick_points(geom_):
    pcd = PointCloud()
    pcd.points = geom_.vertices
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

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

    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath(os.path.join("SMPL_estimator","trained_models", "total_capture_model"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "lsp"


    config = DemoConfig()

    posemodel = YOLO("ultralytics/yolov8l-pose.pt")
    #img_path = "./SMPL_estimator/images_test/slp_25.png"
    #img_path = "./SMPL_estimator/images_test/slp_06.png"
    #img_path = "./SMPL_estimator/images_test/slp_03.jpg"
    #img_path = "./SMPL_estimator/images_test/slp_08.jpg"
    img_path = "./SMPL_estimator/images_test/test_basket2.png"
    img = cv2.imread(img_path)   
    b_img = np.ascontiguousarray(img[:,:,::-1])
    back_img = o3d.geometry.Image(b_img)
    #img = cv2.imread("./SMPL_estimator/images_test/slp_25.png")
    #img = cv2.imread("./SMPL_estimator/images_test/slp_06.png")
    #img = cv2.imread("./SMPL_estimator/images_test/slp_03.jpg")
    #img = cv2.imread("./SMPL_estimator/images_test/slp_08.jpg")
    #img = cv2.imread("./SMPL_estimator/images_test/test_basket4.png")

    H,W = img.shape[0:2] 

    #bbox = [(66,75),(859,531)] # slp_25
    #bbox = [(125,160),(882,536)] # slp_06
    #bbox = [(101,102),(877,515)] # slp_03
    #bbox = [(120,134),(866,470)] # slp_08
    #bbox = [(0,0),(H,H)]
    #bbox = [(64,208),(964,568)]

    ##module = tf.saved_model.load("./MoveNet_thunder")#
    ##img_rgb = np.copy(img[:,:,::-1])
    
    ##p0 = np.array([bbox[0][1],bbox[0][0]]).reshape(-1,2)
    ##person,retimg = movenet_detect(module,img_rgb[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:],input_size=256)
    ##kpts = person.kptsarray + p0

    results = posemodel(img,stream=False)
    kpts = np.float64(results[0].keypoints.data.numpy()[0,:,:])
    print(kpts.shape)
    _sq_img = squarify_img(img,kpts)
    # initialize model
    model = Model()
    original_img, input_img, params,_ = preprocess_image(_sq_img, config.ENCODER_INPUT_SHAPE[0])
    print("PARAMS",params)
    #original_img, input_img, params = preprocess_image(img[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:], config.ENCODER_INPUT_SHAPE[0])
    result = model.detect(input_img)
    cam_params = np.squeeze(result['cam'].numpy())[:3]
    betas = result['shape'].numpy()
    pose = result["pose_shape"].numpy()[0:-10]
    joints = np.squeeze(result["kp2d"].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']

    # Compute the vertices and triangles of the mesh
    vertices = result['vertices'].numpy()
    _, triangles = smpl_mesh_from_params(pose,betas)

    # Load the base mesh containing the uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")

    # Create a Open3D Triangle Mesh
    #_,kpts= get_original(params,vertices,cam_params,joints)
    o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
    o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
    mesh = TriangleMesh(o3d_vertices,o3d_faces)
    mesh.compute_vertex_normals()
    mesh.triangle_uvs = base_mesh.triangle_uvs
    #text_img = cv2.imread("./texture_uv_data/textures/SMPLitex-texture-00000.png")#[:,:,::-1]
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00133.png")
    #text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    geoms = [mesh]
    
    
    Kr = np.array([[2000.0, 0, 0.5*W],
                   [0, 2000.0, 0.5*H],
                   [0, 0, 1]])
    

    Kr244 = np.array([[902.6, 0, 278.4],
                   [0, 877.4, 525.1],
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
    #kpts3d = result["kp3d"].numpy()
    print(kpts3d)
    min_z = np.max(kpts3d[:,2])

    kpts3d += np.array([0.0,0.0,-0.5]).reshape(-1,3)
    
    """
    ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts,Kr,np.array([0,0,0,0]),flags=cv2.SOLVEPNP_SQPNP)
    rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts,Kr,np.array([0,0,0,0]),rvec,tvec,
                                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,100,10*FLT_EPSILON))
    """
    M = get_extrinsics(kpts,kpts3d,Kr)

    #M = np.array([[1,0,0,0],
    #             [0,1,0,0],
    #             [0,0,1,0],
    #             [0,0,0,1]])
    #if ret: print(M)
    


    #img_pose = draw_pose(img,kpts)
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices +np.array([0.0,0.0,-0.5]).reshape(-1,3))
    mesh.compute_vertex_normals()
    vis = o3d.visualization.rendering.OffscreenRenderer(width=W,height=H)
    #vis = o3d.visualization.rendering.O3DVisualizer()
    IntrinsicsCam = o3d.camera.PinholeCameraIntrinsic(W,H,Kr)
    vis.setup_camera(IntrinsicsCam,M)

    material = o3d.visualization.rendering.MaterialRecord()
    material.base_metallic = 0.5  # Non-metallic
    material.base_roughness = 0.6  # High roughness for less gloss
    material.base_reflectance = 0.8  # Medium reflectance
    material.albedo_img = text_img
    material.shader = "defaultLit"

    
    vis.scene.add_geometry("mesh",mesh,material)
    vis.scene.set_background([0.0,0.0,0.0,0.3],image=back_img)
    vis.scene.scene.enable_sun_light(True)
    #print(dir(vis.scene),dir(vis.scene.scene))
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
    #masked = cv2.bitwise_and(rendered_img,rendered_img,mask=mask)+cv2.bitwise_and(img,img,mask=255-mask)
    cv2.imshow("Cropped",_sq_img)
    #mask = np.where(rendered_img[]==np.array([255,255,255],dtype=np.uint8),np.array([255,255,255],dtype=np.uint8),np.array([0,0,0],dtype=np.uint8))
    #cv2.imshow("Overlayed",masked)
    cv2.imshow("With landmarks",draw_pose(img,kpts))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #img_out = draw_prediction_on_image(img, person, crop_reg)
    #o3d.visualization.draw_geometries(geoms)
    