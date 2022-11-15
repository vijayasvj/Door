import yaml 
import face_recognition

emp_image = face_recognition.load_image_file("Sample_pic.jpg")
emp_face_encodings = face_recognition.face_encodings(emp_image)[0]

emp_face_encodings = [emp_face_encodings]
emp_face_names = ["Vijay"]
emp_phno = ["9884581245"]
emp_id = ["01"]

with open(r'YAML encodings/emp_face_encodings.yml', 'w') as f:
    f.write(yaml.dump(emp_face_encodings))

with open(r'YAML encodings/emp_face_names.yml', 'w') as f:
    f.write(yaml.dump(emp_face_names))

with open(r'YAML encodings/emp_phno.yml', 'w') as f:
    f.write(yaml.dump(emp_phno))

with open(r'YAML encodings/emp_id.yml', 'w') as f:
    f.write(yaml.dump(emp_id))