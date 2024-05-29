import numpy as np

class EMA_filter:
    def __init__(self,alphaH,alphaM=None,alphaL=None,vtype="shape") -> None:
        self.last_output = None
        self.alphavec = None
        if vtype == "pose":
            self.vtype = "pose"
            H_idxs = [15]
            M_idxs = [0,1,2,3,6,9,12,13,14]
            L_idxs = [4,5,7,8,10,11,16,17,18,19,20,21,22,23]
            alphavec = []
            for i in range(24):
                if i in H_idxs:
                    alphavec+=[alphaH]*3
                elif i in M_idxs:
                    alphavec+=[alphaM]*3
                elif i in L_idxs:
                    alphavec+=[alphaL]*3
            self.alphavec = np.array(alphavec)
        elif vtype == "shape":
            self.alphavec = np.array([alphaH]*10)
        else:
            self.alphavec = alphaH
    def filter(self,input_vec):
        if self.last_output is None:
            self.last_output = np.copy(input_vec)
            return input_vec
        else:
            self.last_output = (1.0-self.alphavec)*input_vec + self.alphavec*self.last_output
            return self.last_output
        
if __name__ == "__main__":
    print(f"This is the {__file__} file")
