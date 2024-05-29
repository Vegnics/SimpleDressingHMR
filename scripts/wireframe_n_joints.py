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

    # initialize model
    model = Model()
    original_img, input_img, params = preprocess_image("./SMPL_estimator/images_test/tennis_3.png", config.ENCODER_INPUT_SHAPE[0])
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
    text_img = o3d.io.read_image("./texture_uv_data/textures/SMPLitex-texture-00000.png")
    text_img = text_img.flip_vertical()
    mesh.textures = [text_img]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(13776,dtype=np.int32))
    #geoms = [mesh]

    lines = []
    for triangle in triangles:
        lines.append([triangle[0], triangle[1]])
        lines.append([triangle[1], triangle[2]])
        lines.append([triangle[2], triangle[0]])

    line_set = o3d.geometry.LineSet()
    line_set.points = mesh.vertices
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines,dtype=np.int32))

    geoms = [line_set]
    for k,vert in enumerate(smpl_vertices):
        if len(vert)>1:
            v0 = vertices[vert[0]]
            v1 = vertices[vert[1]]
            cnt = np.mean([v0,v1],axis=0)
        else:
            cnt = vertices[vert[0]]
        geoms.append(create_sphere(cnt,0.023))

    """
    kpoints = []
    close_verts = find_closest_vertices(3011,vertices,0.1)
    print(close_verts,len(close_verts))
    close_verts = [2500]
    for vert in close_verts:
        geoms.append(create_sphere(vertices[vert],0.015))
    """
    o3d.visualization.draw_geometries(geoms)
    