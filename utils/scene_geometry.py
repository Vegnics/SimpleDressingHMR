import cv2
import numpy as np
from open3d.geometry import TriangleMesh
import open3d as o3d
import sys
import os
sys.path.append(os.path.abspath('./'))
from utils.postprocessing import EMA_filter

FLT_EPSILON = sys.float_info.epsilon

def create_sphere(center:np.ndarray,radius:float = 0.05):
    sphere = TriangleMesh.create_sphere(radius,50)
    _sphere_verts = np.asarray(sphere.vertices)+np.reshape(center,(1,3))
    sphere.vertices = o3d.utility.Vector3dVector(_sphere_verts)
    sphere.paint_uniform_color(np.array([1.0,0.0,0.0]).reshape(3,1))
    sphere.compute_vertex_normals()
    return sphere

class CameraModel():
    def __init__(self):
        pass
        #self.cam_pose_filter = EMA_filter(0.5,vtype="cam")
        #self.cam_position_filter = EMA_filter(0.45,vtype="cam")
    def get_extrinsics(self,kpts,kpts3d,img_shape,distcoeffs=np.array([0,0,0,0])):
        H,W = img_shape
        kpts2d = kpts[:,0:2][kpts[:,2]>0.7]
        kpts3d = kpts3d[kpts[:,2]>0.7]
        Kr0 = np.array([[7000.0, 0, 0.5*W],
                        [0, 7000.0, 0.5*H],
                        [0, 0, 1]])
        Kr1 = np.array([[100000.0, 0, 0.5*W],
                        [0, 100000.0, 0.5*H],
                        [0, 0, 1]])
        ret,rvec,tvec = cv2.solvePnP(kpts3d,kpts2d,Kr0,distcoeffs,flags=cv2.SOLVEPNP_SQPNP)
        
        #_,tvec = cv2.solvePnPRefineLM(kpts3d,kpts2d,Kr,distcoeffs,rvec,tvec,
        #                                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,60,10*FLT_EPSILON))
        rvec,tvec = cv2.solvePnPRefineLM(kpts3d,kpts2d,Kr1,distcoeffs,rvec,tvec,
                                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,80,10*FLT_EPSILON))
        _tvec = np.vstack([tvec.reshape(3,1),np.array([[1.0]])]).reshape(4,1) 
        _rmat = np.vstack([cv2.Rodrigues(rvec)[0],np.zeros((1,3))])
        return np.hstack([_rmat,_tvec])

if __name__ == "__main__":
    print(f"This is the {__file__} file")