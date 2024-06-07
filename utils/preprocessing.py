import numpy as np
import cv2


def optimal_crop_from_kpts(img,kpnts):
    kpnt_val = [0,1,2,3,4,5,6,7,8,11,12,13,14,15,16]
    _kpnts = kpnts[kpnt_val,:]
    _kpnts = _kpnts[:,0:2][_kpnts[:,2]>0.65]
    _size_body = 140 #145
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
        x_min = int(c_hip[0]-bbox_h/2-rem*scale)
        x_max = int(c_hip[0]+bbox_h/2+rem*scale)
    else:
        x_min = int(c_hip[0]-bbox_w/2-rem*scale)
        x_max = int(c_hip[0]+bbox_w/2+rem*scale)
        y_min = int(c_hip[1]-bbox_w/2-rem*scale)
        y_max = int(c_hip[1]+bbox_w/2+rem*scale)

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
    return cv2.copyMakeBorder(img[y_min:y_max,x_min:x_max,:],top,bottom,left,right,cv2.BORDER_REPLICATE)
    #return np.copy(img[y_min:y_max,x_min:x_max,:])

def HMR_preprocess_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 1.
    if np.max(img.shape[:2]) != img_size:
        print('Resizing image to {}'.format(img_size))
        scale = (float(img_size) / np.max(img.shape[:2]))

    image_scaled, actual_factor = resize_img(img, scale)
    center = np.round(np.array(image_scaled.shape[:2]) / 2).astype(int)
    center = center[::-1]  # image center in (x,y)

    margin = int(img_size / 2)
    image_pad = np.pad(image_scaled, ((margin,), (margin,), (0,)), mode='edge')
    center_pad = center + margin
    start = center_pad - margin
    end = center_pad + margin

    crop = image_pad[start[1]:end[1], start[0]:end[0], :]
    crop = 2 * ((crop / 255.) - 0.5)  # Normalize image to [-1, 1]

    params = {'img_size': img_size, 'scale': scale, 'start': start, 'end': end, }

    return img, crop, params


def resize_img(img, scale):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])]
    return new_img, actual_factor
    