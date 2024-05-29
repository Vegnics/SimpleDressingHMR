import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

from config import Config
from model import Model
#from trimesh_renderer import TrimeshRenderer
from vis_util import preprocess_image#, visualize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        LOG_DIR = os.path.abspath('./{}/{}'.format("trained_models", "total_capture_model"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "lsp"


    config = DemoConfig()

    # initialize model
    model = Model()
    original_img, input_img, params = preprocess_image("images_test/cheering.png", config.ENCODER_INPUT_SHAPE[0])

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']

    print(cam,joints)
    print("SHAPE")
    print(result['shape'].numpy())
    print("POSE_shape")
    pose_lst = list(result["pose_shape"].numpy()[0:-10])
    str_pose = ""
    for i,val in enumerate(pose_lst):
        str_pose += "{}, ".format(val)
        if (i+1)%3==0:
            str_pose += "\n"
    print(str_pose)
    #for _mat in result['pose'].numpy():
    #    r = R.from_matrix(_mat)
    #    print(r.as_mrp())
    #renderer = TrimeshRenderer()
    #visualize(renderer, original_img, params, vertices, cam, joints)
