import streamlit as st
import requests
from PIL import Image
import yaml
import face_recognition
import os
import numpy as np
import cv2


def load_image(image_file):
	img = Image.open(image_file)
	return img

image = []
zh =1
st.set_page_config(layout = "wide")

body = st.container()

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_seidgi4z.json")
img = Image.open(r"orgqare_logo.png")

with body:
    st.title('Update your info in Orgware’s server')
    st.write('##')
    #st.write('##')

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("Welcome to ORGware Technologies, a client centric global Mobility software development company providing Mobile apps development, Web application development , research, web development, IT outsourcing services along with solutions and consulting services for your mission-critical business challenges.")        

    with right_column:
        st.image(img)
    #st.write("---")

    image = st.file_uploader("Choose a file", type=["png","jpg","jpeg"], accept_multiple_files=False, key=None, help="content image")
    name = st.text_input("Enter your name")
    phno = st.text_input("Enter your phone number")
    empid = st.text_input("Enter your employee ID")
    submitted = st.button("Submit your personal info")

    if submitted:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        with open(r'YAML encodings/emp_face_encodings.yml', 'r') as f:
            emp_face_encodings = yaml.load(f.read(), Loader=yaml.Loader)

        with open(r'YAML encodings/emp_face_names.yml', 'r') as f:
            emp_face_names = yaml.load(f.read(), Loader=yaml.Loader)

        with open(r'YAML encodings/emp_phno.yml', 'r') as f:
            emp_phno = yaml.load(f.read(), Loader=yaml.Loader)

        with open(r'YAML encodings/emp_id.yml', 'r') as f:
            emp_id = yaml.load(f.read(), Loader=yaml.Loader)

        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        emp_face_encodings.extend(face_encodings)
        emp_face_names.append(name)
        emp_phno.append(phno)
        emp_id.append(empid)

        os.remove(r'YAML encodings/emp_face_encodings.yml')
        os.remove(r'YAML encodings/emp_face_names.yml')
        os.remove(r'YAML encodings/emp_phno.yml')
        os.remove(r'YAML encodings/emp_id.yml')

        with open(r'YAML encodings/emp_face_encodings.yml', 'w') as f:
            f.write(yaml.dump(emp_face_encodings))

        with open(r'YAML encodings/emp_face_names.yml', 'w') as f:
            f.write(yaml.dump(emp_face_names))

        with open(r'YAML encodings/emp_phno.yml', 'w') as f:
            f.write(yaml.dump(emp_phno))

        with open(r'YAML encodings/emp_id.yml', 'w') as f:
            f.write(yaml.dump(emp_id))


        
 

