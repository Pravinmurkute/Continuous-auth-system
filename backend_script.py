import face_recognition

known_image = face_recognition.load_image_file(r"D:\Continuous_auth_system\dataset\known_faces\pravin.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

test_image = face_recognition.load_image_file(r"D:\Continuous_auth_system\dataset\known_faces\pravin.jpg")  # Replace with an actual image path
test_encoding = face_recognition.face_encodings(test_image)

if test_encoding:
    result = face_recognition.compare_faces([known_encoding], test_encoding[0])
    print("Authentication Result:", "Success" if result[0] else "Fail")
else:
    print("No face detected in test image!")
