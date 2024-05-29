import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from ld_data import smpl_vertices

# to make run from console for module import
sys.path.append(os.path.abspath('./'))

from SMPL_estimator.config import Config
from SMPL_estimator.model import Model
from SMPL_estimator.vis_util import preprocess_image
from smpl.compute_smpl_mesh import smpl_mesh_from_params
import open3d as o3d
from open3d.geometry import TriangleMesh,PointCloud
from typing import Tuple
import tensorflow as tf
from Movenet_core import _personLandmarkResults
from Movenet_drawing import draw_prediction_on_image
import cv2

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
    _size_body = 180
    rem = (224-_size_body)/2
    H,W = img.shape[0:2]
    bbox_x0 = np.min(kpnts[:,0]) 
    bbox_y0 = np.min(kpnts[:,1])
    bbox_x1 = np.max(kpnts[:,0])
    bbox_y1 = np.max(kpnts[:,1])
    bbox_h = bbox_y1-bbox_y0
    bbox_w = bbox_x1-bbox_x0
    c_hip = 0.5*(kpnts[11,:]+kpnts[12,:])
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
    return cv2.resize(cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE),
                      (224,224),cv2.INTER_LINEAR)
    #return cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE)

def get_extrinsics(rvec,tvec):
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
    if not retPerson:
        return keypoints_with_scores
    else:
        person = _personLandmarkResults()
        person.fromTensor(keypoints_with_scores, img_rgb.shape)
        return person




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
    img = cv2.imread("./SMPL_estimator/images_test/test_basket.png")

    img_rgb = np.copy(img[:,:,::-1])
    # initialize model
    model = Model()
    original_img, input_img, params,_ = preprocess_image(img_rgb, config.ENCODER_INPUT_SHAPE[0])
    #original_img, input_img, params = preprocess_image(img[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1],:], config.ENCODER_INPUT_SHAPE[0])
    result = model.detect(input_img)
    betas = result['shape'].numpy()
    pose = result["pose_shape"].numpy()[0:-10]

    # Compute the vertices and triangles of the mesh
    vertices, triangles = smpl_mesh_from_params(pose,betas)

    # Load the base mesh containing the uv maps
    base_mesh = o3d.io.read_triangle_mesh("./texture_uv_data/smpl_uv.obj")

    # Create a Open3D Triangle Mesh
    o3d_vertices = o3d.utility.Vector3dVector(np.copy(vertices))
    o3d_faces = o3d.utility.Vector3iVector(np.copy(triangles))
    mesh = TriangleMesh(o3d_vertices,o3d_faces)
    mesh.compute_vertex_normals()
    mesh.triangle_uvs = base_mesh.triangle_uvs
    #text_img = cv2.imread("./texture_uv_data/textures/SMPLitex-texture-00000.png")#[:,:,::-1]
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00133.png")
    text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    geoms = [mesh]
    
    o3d.visualization.draw_geometries(geoms)
    