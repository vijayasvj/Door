from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import face_recognition
import streamlit as st
import numpy as np
import yaml
from collections import defaultdict
from PIL import Image
import os
import requests
from datetime import datetime
import time


IMG_SIZE = 24

#eyes_detected = defaultdict(str)
face_cascPath = 'haarcascade_frontalface_alt.xml'
open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
right_eye_cascPath ='haarcascade_righteye_2splits.xml'
face_detector = cv2.CascadeClassifier(face_cascPath)
open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)
#model = load_model()


with open(r'YAML encodings/emp_face_encodings.yml', 'r') as f:
    emp_face_encodings = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_face_names.yml', 'r') as f:
    emp_face_names = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_phno.yml', 'r') as f:
    emp_phno = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_id.yml', 'r') as f:
    emp_id = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_enter_details.yml', 'r') as f:
    emp_enter_details = yaml.load(f.read(), Loader=yaml.Loader)

class VideoProcessor:
	def recv(self, frame):
		frame = frame.to_ndarray(format="bgr24")
		frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
		faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # for each detected face
		for (x,y,w,h) in faces:
            # Encode the face into a 128-d embeddings vector
			encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]
            # Compare the vector with all known faces encodings
			matches = face_recognition.compare_faces(emp_face_encodings, encoding)
            # For now we don't know the person name
			name = "Unknown"
            # If there is at least one match:
			if True in matches:
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				for i in matchedIdxs:
					name = emp_face_names[i]
					counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of votes
				name = max(counts, key=counts.get)
				now = datetime.now()
				current_time = now.strftime("%H:%M:%S")
				text = name + "entered the organization at " + current_time
				emp_enter_details.append(text)
				os.remove(r'YAML encodings/emp_enter_details.yml')
				with open(r'YAML encodings/emp_enter_details.yml', 'w') as f:
					f.write(yaml.dump(emp_enter_details))
				requests.get("http://10.10.103.209/face")
				face = frame[y:y+h,x:x+w]
				gray_face = gray[y:y+h,x:x+w]
			cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
			
		return av.VideoFrame.from_ndarray(frame, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)

