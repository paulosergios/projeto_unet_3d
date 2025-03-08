import os
import numpy as np
from script.predict import predict_3d
from script.predict import load_nifti_image
from visualization import visualize_results
from keras.models import load_model

SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def main():
    # Caminhos para as imagens e o modelo
    image_path = "C:/Users/paulo/Desktop/Dataset/database/testing/patient111/patient111_frame01.nii.gz"
    output_path = "C:/Users/paulo/Desktop/predict.nii.gz"
    model_weights_path = "C:/Users/paulo/Desktop/projeto_unet_para_3d/models/best_unet_model.keras"
    true_mask_path = "C:/Users/paulo/Desktop/Dataset/database/testing/patient111/patient111_frame01_gt.nii.gz"

    # Carregando o modelo
    try:
        model = load_model(model_weights_path)
    except Exception as e:
        return

    # Realizando a predição
    try:
        predict_3d(model, image_path, output_path, true_mask_path)
    except Exception as e:
        return
    
    # Carregando a imagem original
    try:
        original_image = load_nifti_image(image_path)
    except Exception as e:
        return

    # Carregando a máscara verdadeira, se existir
    true_mask = None
    if os.path.exists(true_mask_path):
        try:
            true_mask = load_nifti_image(true_mask_path)
        except Exception as e:
            return
    
    # Visualizando os resultados
    if os.path.exists(output_path):
        try:
            visualize_results(original_image, output_path, true_mask)
        except Exception as e:
            return

if __name__ == "__main__":
    main()
