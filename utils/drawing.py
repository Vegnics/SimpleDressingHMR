import cv2
import numpy as np

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
    for pnt in kpnts:
        cv2.circle(_img,(int(pnt[0]),int(pnt[1])),7,(0,0,255),-1)
    return _img