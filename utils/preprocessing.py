import numpy as np


def optimal_crop_from_kpts(img,kpnts):
    _kpnts = kpnts[:,0:2][kpnts[:,2]>0.6]
    _size_body = 145
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
    else:
        x_min = int(c_hip[0]-bbox_w/2-rem*scale)
        x_max = int(c_hip[0]+bbox_w/2+rem*scale)
        y_min = int(c_hip[1]-bbox_w/2-rem*scale)#int(bbox_y0)#
        y_max = int(c_hip[1]+bbox_w/2+rem*scale)#int(bbox_y1)#

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
    #print("padding ",top,bottom,left,right)
    #return cv2.resize(cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE),
    #                  (224,224),cv2.INTER_LINEAR)
    #return cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE)
    return np.copy(img[y_min:y_max,x_min:x_max,:])
    