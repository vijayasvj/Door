import cv2
import sys

video_capture = cv2.VideoCapture(0)


import av
import cv2
import face_recognition
import streamlit as st
import numpy as np
import yaml
from collections import defaultdict
from PIL import Image
from keras.models import model_from_json
from skimage.transform import resize

IMG_SIZE = 24

def load_model():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return loaded_model

def predict(img, model):
	img = Image.fromarray(img, 'RGB').convert('L')
	img = resize(img, (IMG_SIZE,IMG_SIZE)).astype('float32')
	img /= 255
	img = img.reshape(1,IMG_SIZE,IMG_SIZE,1)
	prediction = model.predict(img)
	if prediction < 0.1:
		prediction = 'closed'
	elif prediction > 0.9:
		prediction = 'open'
	else:
		prediction = 'idk'
	return prediction

def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

eyes_detected = defaultdict(str)
face_cascPath = 'haarcascade_frontalface_alt.xml'
open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
right_eye_cascPath ='haarcascade_righteye_2splits.xml'
face_detector = cv2.CascadeClassifier(face_cascPath)
open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)
model = load_model()


with open(r'YAML encodings/emp_face_encodings.yml', 'r') as f:
    emp_face_encodings = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_face_names.yml', 'r') as f:
    emp_face_names = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_phno.yml', 'r') as f:
    emp_phno = yaml.load(f.read(), Loader=yaml.Loader)

with open(r'YAML encodings/emp_id.yml', 'r') as f:
    emp_id = yaml.load(f.read(), Loader=yaml.Loader)


def recv(frame):
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
            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]
        eyes = []
        # Eyes detection
        # check first if eyes are open (with glasses taking into account)
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        # if open_eyes_glasses detect eyes then they are open 
        if len(open_eyes_glasses) == 2:
            eyes_detected[name]+='1'
            for (ex,ey,ew,eh) in open_eyes_glasses:
                cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # otherwise try detecting eyes using left and right_eye_detector
        # which can detect open and closed eyes                
        else:
            # separate the face into left and right sides
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]
            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]
            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            eye_status = '1' # we suppose the eyes are open
            # For each eye check wether the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex,ey,ew,eh) in right_eye:
                color = (0,255,0)
                pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
            for (ex,ey,ew,eh) in left_eye:
                color = (0,255,0)
                pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
            eyes_detected[name] += eye_status
        # Each time, we check if the person has blinked
        # If yes, we display its name
        if isBlinking(eyes_detected[name],3):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display name
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        print(name)
    return frame

while True:
    ret, frame = video_capture.read()
    frame = recv(frame)
    cv2.imshow('Intha vaainko', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()