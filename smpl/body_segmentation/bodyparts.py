import json
import open3d as o3d
import numpy as np

with open("./smpl/body_segmentation/smpl_vert_segmentation.json","r") as file:
    bodyvertices = json.load(file)

mesh = o3d.io.read_triangle_mesh("/home/amaranth/Desktop/NTU_2024/Computer_graphics/Final_project/simple_smpl/hello_smpl.obj")

"""
['rightHand', 
'rightUpLeg', 
'leftArm', 
'leftLeg', 
'leftToeBase', 
'leftFoot', 
'spine1', 
'spine2', 
'leftShoulder', 
'rightShoulder', 'rightFoot', 'head', 'rightArm', 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', 'leftUpLeg', 'leftHand', 'hips']
"""

bodyparts = ["head",
             "neck",
             "rightToeBase",
             "leftToeBase",
             #"rightHand",
             #"leftHand",
             #'rightForeArm',
             #'leftForeArm',
             "rightFoot",
             "leftFoot"]

data = {}
data["triangles"] = []

target_vertices = []
for part in bodyparts:
    target_vertices+=bodyvertices[part]

print(len(target_vertices))

in_triangles = np.asarray(mesh.triangles)
for idx,tri in enumerate(in_triangles):
    if tri[0] in target_vertices and tri[1] in target_vertices and tri[2] in target_vertices:
        data["triangles"].append(idx)


with open("./smpl/body_segmentation/removed_triangles.json","w") as file:
    json.dump(data,file)


