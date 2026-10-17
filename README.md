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


adding new feature-(recognize the objects in real time)-ultralytics yolo
python -m pip install ultralytics
to check if it is compatable
python -c "import cv2; print(cv2.__version__); print('Face module:', hasattr(cv2, 'face'))"


new fewature -Air writing
python -m pip install mediapipe
python air_writing.py
