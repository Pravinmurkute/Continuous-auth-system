import bz2
import shutil

# Set paths
bz2_file = "D:\\Continuous_auth_system\\models\\dlib_face_recognition_resnet_model_v1.dat.bz2"
output_file = "D:\\Continuous_auth_system\\models\\dlib_face_recognition_resnet_model_v1.dat"

# Extract .bz2 file
with bz2.BZ2File(bz2_file) as f_in, open(output_file, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

print("✅ Extraction completed!")
