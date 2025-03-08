from script.predict import predict_3d
from keras.models import load_model

image_path = "C:/Users/paulo/Desktop/Dataset/database/testing/patient110/patient110_frame01.nii.gz"
output_path = "C:/Users/paulo/Desktop/predict.nii.gz"
model_weights_path = "C:/Users/paulo/Desktop/projeto_unet/models/unet/best_unet_model.keras"

model = load_model(model_weights_path)

predict_3d(model, image_path, output_path)