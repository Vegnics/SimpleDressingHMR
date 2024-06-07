# Combining Human Mesh Recovery (HMR), skeleton-based animation and texture mapping for simple dressing simulation

[![Python 3.10](https://img.shields.io/badge/python-+3.10-blue.svg)]()
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](LICENSE)

Project description:
========
The present repository contains the term project for **Interactive Computer Graphics (ICG)** course during the spring semester 2024 at **National Taiwan University (NTU)**.

<img src="./imgs/results_1.gif" height="400">

The presented project aims to simulate dressing with different styles with Open3D (Python API docs: https://www.open3d.org/docs/release/) as the rendering engine. The Skinned Multi Person Linear (SMPL) model is adopted to describe the body mesh of a subject (male or female). In order to estimate the SMPL parameters (shape, pose) we are using the Human Mesh Recovery (HMR) approach proposed by Kanazawa et al. (2018) <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.html" alt="Releases"><img src="https://img.shields.io/badge/[paper]-8A2BE2"/></a>. Given that the original implementation of HMR <a href="https://github.com/akanazawa/hmr" alt="Forks"><img src="imgs/github.png" alt="drawing" width=20/></a> requires Tensorflow 1.x, we decided to utilize the reimplementation by Alessandro Russo <a href="https://github.com/russoale/hmr2.0" alt="Forks"><img src="imgs/github.png" alt="drawing" width=20/></a> (Tensorflow 2.x is used). 

Given that HMR have some requirements to work properly. The input image must have a shape of 224x224 px,the body lenght must be around 150 px, and the hip center must be in the middle of the input image. If these conditions are not met, the resulting mesh might be innacurate. In order to meet these requirements, we are employing the pre-trained Human Pose Estimation model offered by Ultralytics (YOLOv8-pose). Using the detected body landmarks we compute the correct bounding box, and keep the 2D landmark locations for the following steps.  After the parameters are obtained with HMR (pose and shape), SMPL is employed to generate a 3D mesh. Then, 2D landmarks and 3D landmarks (obtained from the 3D mesh) are used in conjunction with a PnP algorithm to compute the extrinsic parameters of the virtual camera (we modeled an Orthographic projection camera as a pinhole camera with a very large focal length). Texture mapping is applied to the 3D mesh, and some triangle faces are removed (corresponding to the head, neck, and feet). Finally the mesh is rendered with Open3D, using the input image as the background image.   


To learn about SMPL, please visit the website: http://smpl.is.tue.mpg
You can find the SMPL paper at: http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

The original SMPL models can be found at: http://smpl.is.tue.mpg/downloads

More Human Pose Estimation models can be downloaded from the Ultralytics website: https://docs.ultralytics.com/tasks/pose/

Check the repo about HMR (*Domain Analysis of End-to-end Recovery of Human Shape and Pose*), from Alessandro Russo, at 

The textures employed in this project were taken from the **SMPLitex** project. Check its website to download more interesting textures: https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html
 



**Team members**
- 林保羅 (Paulo Linares) -- D12922028
- 劉容綺 -- R11528025

For comments or questions, please email us at: d12922028@ntu.edu.tw



System Requirements:
====================
Operating system: Linux (Ubuntu 20.04 is preferred)

Python version: 3.10.0

Python Dependencies:
- OpenCV		: 4.9.0.80
- Scipy			: 1.13.0
- iopath		: 0.1.10
- Tensorflow	: 2.10.0
- Numpy			: 1.23.5
- Chumpy		: 0.70.0
- Imageio		: 2.34.1
- Ultralytics	: 8.1.0
- Open3D		: 0.18.0
- Torch			: 1.13.0


Getting Started:
================

1. Clone the repository:
--------------------
	git clone https://github.com/Vegnics/SimpleDressingHMR.git


2. Create a Conda environment:
----------------------
	conda create -n icg24 python=3.10.0
	conda activate icg24

3. Download (install) the pre-trained models
----------------------
	chmod +x ./get_models.sh
	bash ./get_models.sh

4. Install the python library dependencies
----------------------
	python -m pip install -r requirements.txt

Running the scripts
=======================
###  Run the demo script for dressing simulation from a video (_*.mp4_ is preferable)
------------------------
	python scripts/video2texmesh.py --model MODEL --ivideo "NAME_OF_THE_VIDEO.mp4" --ovideo "OUTPUT_VIDEO_NAME.mp4" --textimg "TEXTURE_NAME.png"

* --model: Name of the SMPL model: *male*, *female*, or  *neutral* (it is recommended to choose the *male* option)

* --ivideo: Name of the input video to be processed (should be inside the _videos/_  folder). E.g. *"videocap001.mp4"*

* --ovideo: Name of the resulting video (it will be saved in the _videos/_ folder) E.g. *"processedvideo.mp4"*

* --textimg: Name of the image containing the texture to be applied to the 3D mesh (must be inside the *texture_uv_data/textures/* folder) E.g *"SMPLitex-texture-00078.png"*

	* Example: 
	------------	
		python scripts/video2texmesh.py --model male --ivideo "videocap001.mp4" --ovideo "test_001.mp4" --textimg SMPLitex-texture-00059.png


###  Run the demo script for dressing simulation from an image
-------------------
	python scripts/img2texmesh.py --model MODEL --image "IMAGE_NAME.xxx" --textimg "TEXTURE_NAME.png"

* --model: 	Name of the SMPL model: *male*, *female*, or  *neutral* (it is recommended to choose the *male* option)

* --image: Name of the image to be processed (must contain only one person, and must placed in the *images_test/* folder). E.g. *cheering_L.png*

* --textimg: Name of the image containing the texture to be applied to the 3D mesh (must be inside *texture_uv_data/textures/* ) E.g *"SMPLitex-texture-00078.png"*

	* Example:
	------------
		python scripts/img2texmesh.py --model female --image "cheering_L.png" --textimg "SMPLitex-texture-00059.png"

###  Run the demo script for visualizing (interactive Open3D visualizer) a textured SMPL mesh on from an image
-------------------

	python scripts/visualize_from_img.py --model neutral --image "cheering_L.png" --textimg "SMPLitex-texture-00193.png"

Use the mouse to rotate/shift the 3D mesh. Press "S" to get a smoother surface.

--------------------------------
### Note:
All of the scripts in _./scripts_  will require the dependencies listed above. The scripts are provided as a sample to help you get started. 

