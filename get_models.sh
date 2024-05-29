# The models employed in this work are implemented in Tensorflow and Torch so 
# they are not included in the repository. You can run this Shell script to download
# and place the models in the right folder.  

# Ultralytics Human Pose Estimation model 
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/yolov8l-pose.pt
mv yolov8l-pose.pt ./ultraytics/.

#-----------------------------------------------------------------------------------

# SMPL models (female, male, neutral | 10 PCs)

# Female
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/basicmodel_f_lbs_10_207_0_v1.0.0.pkl
mv ./basicmodel_f_lbs_10_207_0_v1.0.0.pkl ./smpl/models/smpl/.

# Male
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
mv ./basicmodel_m_lbs_10_207_0_v1.0.0.pkl ./smpl/models/smpl/.

# Neutral
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
mv ./basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ./smpl/models/smpl/.


#-----------------------------------------------------------------------------------

# HMR pre-trained models

# Total capture - paired (most accurate)
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/total_capture_model_paired.zip
mv ./total_capture_model_paired.zip ./SMPL_estimator/trained_models/.
unzip ./SMPL_estimator/trained_models/total_capture_model_paired.zip -d ./SMPL_estimator/trained_models/.
rm -f ./SMPL_estimator/trained_models/total_capture_model_paired.zip

# Total capture - unpaired (works well when the person is not turning in the video)
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/total_capture_model.zip
mv ./total_capture_model.zip ./SMPL_estimator/trained_models/.
unzip ./SMPL_estimator/trained_models/total_capture_model.zip -d ./SMPL_estimator/trained_models/.
rm -f ./SMPL_estimator/trained_models/total_capture_model.zip

# Base (the shape parameters are not very accurate, unacceptable performance for the purpose of this project)
wget https://github.com/Vegnics/SimpleDressingHMR/releases/download/v0.0.1/base_model.zip
mv ./base_model.zip ./SMPL_estimator/trained_models/.
unzip ./SMPL_estimator/trained_models/base_model.zip -d ./SMPL_estimator/trained_models/.
rm -f ./SMPL_estimator/trained_models/base_model.zip


