import numpy as np

class EMA_filter:
    def __init__(self,alphaH,alphaM=None,alphaL=None,vtype="shape") -> None:
        self.last_output = None
        self.alphavec = None
        self.vtype = vtype
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
            #self.alphavec = np.array([i*alphaH for i in [1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91]])
            self.alphavec = np.array([i*alphaH for i in [1.0,0.96,0.94,0.92,0.91,0.908,0.906,0.904,0.902,0.9]])
        else:
            self.alphavec = alphaH
    def filter(self,input_vec):
        if self.last_output is None:
            self.last_output = np.copy(input_vec)
            return self.last_output
        else:
            self.last_output = (1.0-self.alphavec)*input_vec + self.alphavec*self.last_output
            #_out_val = (1.0-self.alphavec)*input_vec + self.alphavec*self.last_output
            #error = np.mean(np.abs(self.last_output - _out_val))
            #if (self.vtype == "pose" and error <0.014) or (self.vtype == "shape" and error <0.04):
            #    self.last_output = (1.0-self.alphavec)*input_vec + self.alphavec*self.last_output
            #else:
            #    self.last_output = 0.25*(1.0-self.alphavec)*input_vec + 0.75*self.alphavec*self.last_output
            #self.l
            #print(self.vtype,error)
            return self.last_output
        
if __name__ == "__main__":
    print(f"This is the {__file__} file")
