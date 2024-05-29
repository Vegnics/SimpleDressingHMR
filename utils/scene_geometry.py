import cv2
import numpy as np
from open3d.geometry import TriangleMesh
import open3d as o3d
import sys

FLT_EPSILON = sys.float_info.epsilon

def create_sphere(center:np.ndarray,radius:float = 0.05):
    sphere = TriangleMesh.create_sphere(radius,50)
    _sphere_verts = np.asarray(sphere.vertices)+np.reshape(center,(1,3))
    sphere.vertices = o3d.utility.Vector3dVector(_sphere_verts)
    sphere.paint_uniform_color(np.array([1.0,0.0,0.0]).reshape(3,1))
    sphere.compute_vertex_normals()
    return sphere

def get_extrinsics(kpts,kpts3d,Kr,distcoeffs=np.array([0,0,0,0])):
    kpts2d = kpts[:,0:2][kpts[:,2]>0.6]
    kpts3d = kpts3d[kpts[:,2]>0.6]
    ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts2d,Kr,distcoeffs,flags=cv2.SOLVEPNP_SQPNP)
    rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts2d,Kr,distcoeffs,rvec,tvec,
                                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,100,10*FLT_EPSILON))
    _tvec = np.vstack([tvec,np.array([[1.0]])]).reshape(4,1) 
    _rmat = np.vstack([cv2.Rodrigues(rvec)[0],np.zeros((1,3))])
    return np.hstack([_rmat,_tvec])

if __name__ == "__main__":
    print(f"This is the {__file__} file")