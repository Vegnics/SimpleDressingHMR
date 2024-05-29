# The models employed in this work are implemented in Tensorflow and Torch so 
# they are not included in the repository. You can run this Shell script to download
# and place the models in the right folder.  

# Ultralytics Human Pose Estimation model 
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt
mv yolov8l-pose.pt ./ultraytics/.

#-----------------------------------------------------------------------------------

# SMPL models (female, male, neutral | 10 PCs)
# Male
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S6Twwbk5QrfZM36bJKU0uNlg5W6OCJC6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1S6Twwbk5QrfZM36bJKU0uNlg5W6OCJC6" -O basicmodel_f_lbs_10_207_0_v1.0.0.pkl && rm -rf /tmp/cookies.txt
mv ./basicmodel_f_lbs_10_207_0_v1.0.0.pkl ./smpl/models/smpl/.

# Female
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FRrC8wzBVlOX77Rq_WdPrB4RoDaoZ-1L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FRrC8wzBVlOX77Rq_WdPrB4RoDaoZ-1L" -O basicmodel_m_lbs_10_207_0_v1.0.0.pkl && rm -rf /tmp/cookies.txt
mv ./basicmodel_m_lbs_10_207_0_v1.0.0.pkl ./smpl/models/smpl/.

# Neutral
curl -H "Authorization: Bearer ya29.a0AXooCgvavov45Kqx0jmYslAABk2wbVTpAJajIfruzOwc70DdY3h5RkpErsY56-2O3qwZ-lSMJiNc1gYnIlUqUnW5EBxtjTpnT4byZ_uxOVApB1IjbkcFYBF9Xzc7igsz-pkwyUOlNGFtIHM4hCJpirgrUeID5rlRy_vCaCgYKASQSARISFQHGX2MiRSrfhyBRAw2zNZFerY9k8A0171" https://www.googleapis.com/drive/v3/files/19IUy6DVci_CSIK7msvcfpjOD4AT_dIHy?alt=media -o basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
mv ./basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ./smpl/models/smpl/.


#-----------------------------------------------------------------------------------

# HMR pre-trained models

# Total capture - paired (most accurate)
curl -H "Authorization: Bearer ya29.a0AXooCgvavov45Kqx0jmYslAABk2wbVTpAJajIfruzOwc70DdY3h5RkpErsY56-2O3qwZ-lSMJiNc1gYnIlUqUnW5EBxtjTpnT4byZ_uxOVApB1IjbkcFYBF9Xzc7igsz-pkwyUOlNGFtIHM4hCJpirgrUeID5rlRy_vCaCgYKASQSARISFQHGX2MiRSrfhyBRAw2zNZFerY9k8A0171" https://www.googleapis.com/drive/v3/files/10j4t0_F3rkcO0pBWsXdE_D402CiYG7-h?alt=media -o total_capture_model_paired.zip
mv ./total_capture_model_paired.zip ./SMPL_estimator/trained_models/.
unzip ./SMPL_estimator/trained_models/total_capture_model_paired.zip -d ./SMPL_estimator/trained_models/.
rm -f ./SMPL_estimator/trained_models/total_capture_model_paired.zip

