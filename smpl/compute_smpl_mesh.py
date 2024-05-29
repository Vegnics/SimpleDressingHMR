import sys
import os
print(os.getcwd())
sys.path.insert(1,"/home/amaranth/Desktop/NTU_2024/Computer_graphics/Final_project/simple_smpl/smpl/smpl_webuser")
from smpl.serialization import load_model
import pickle
import numpy as np
from typing import Tuple

class SMPL_mesh:
    def __init__(self,model_type="neutral"):
        self.model = None
        if model_type == "neutral":
            self.model = load_model('smpl/models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
        elif model_type == "male":
            self.model = load_model('smpl/models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        elif model_type == "female":
            self.model = load_model('smpl/models/smpl/basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
    def get_mesh_from_params(self,pose: np.ndarray,
                            betas: np.ndarray)-> Tuple[np.ndarray,np.ndarray]:
        #m = load_model('smpl/models/smpl/basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        self.model.pose[:] = pose
        self.model.betas[0:10] = betas
        return self.model.r,self.model.f

if __name__ == '__main__':
    print(__name__)
