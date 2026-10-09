first steup virtual ebvironment 

python -m venv venv

then activate it by 

.\venv\Scripts\Activate.ps1

install opencv for facial recognization feature 
pip install opencv-contrib-python

(version 4.something)


FOR GET AND TRAIN DATASET
python capture_faces.py
(enter name)
TRAIN IT 
python train.py
THEN RUN IT
python recognize.py


Adding new feature of emotion 
pip install deepface
python emotion.py

