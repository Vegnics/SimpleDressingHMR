import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath('./'))

from SMPL_estimator_SPIN.hmr import get_hmr
from SMPL_estimator_SPIN.imutils import crop
#from utils.renderer import Renderer
import SMPL_estimator_SPIN.config as config
import SMPL_estimator_SPIN.constants as constants
from smpl.compute_smpl_mesh import smpl_mesh_from_params
import open3d as o3d
from open3d.geometry import TriangleMesh,PointCloud
from typing import Tuple

"""
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

"""


def process_image(img_rgb, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    #img = cv2.imread(img_rgb)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    img = img_rgb[:,:,::-1].copy()

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200
    """
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    """
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    #args = parser.parse_args()
    chkpnt_path = "/home/amaranth/Desktop/NTU_2024/Computer_graphics/Final_project/simple_smpl/SMPL_estimator_SPIN/trained_models/model_checkpoint.pt"
    img_path = "/home/amaranth/Desktop/NTU_2024/Computer_graphics/Final_project/simple_smpl/SMPL_estimator/images_test/cheering.png"

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    device = torch.device('cpu')
    # Load pretrained model
    model = get_hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(chkpnt_path, map_location=torch.device('cpu'))
    #print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Preprocess input image and generate predictions
    img_in = cv2.imread(img_path)
    img, norm_img = process_image(img_in, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        #pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        #pred_vertices = pred_output.vertices
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()

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
    mesh.compute_vertex_normals()
    mesh.triangle_uvs = base_mesh.triangle_uvs
    #text_img = cv2.imread("./texture_uv_data/textures/SMPLitex-texture-00000.png")#[:,:,::-1]
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00133.png")
    text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    geoms = [mesh]
    
    o3d.visualization.draw_geometries(geoms)