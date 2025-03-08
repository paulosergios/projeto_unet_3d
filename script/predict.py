import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage import morphology
import os

def load_nifti_image(filepath):
    """Carrega uma imagem NIfTI e retorna seus dados."""
    print(f"Carregando imagem NIfTI de {filepath}...")
    nifti_image = nib.load(filepath)
    data = np.array(nifti_image.get_fdata())
    print(f"Imagem carregada com shape: {data.shape}")
    return data

def save_nifti_image(filepath, image_data):
    """Salva uma imagem NIfTI no caminho especificado."""
    print(f"Tentando salvar a imagem NIfTI em {filepath}...")
    nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
    nib.save(nifti_img, filepath)
    print(f"Imagem salva com sucesso em {filepath}!")

def smooth_mask(mask, sigma=1):
    """Aplica suavização na máscara."""
    print("Aplicando suavização na máscara...")
    return gaussian_filter(mask, sigma=sigma)

def postprocess_mask(mask, percentile=90):
    """Processa a máscara: suaviza, binariza e remove objetos pequenos."""
    print("Processando a máscara predita...")
    smoothed_mask = smooth_mask(mask)
    threshold_value = 0.1
    print(f"Valor de limiar calculado para binarização: {threshold_value}")
    binary_mask = (smoothed_mask > threshold_value).astype(np.uint8)
    print("Removendo objetos pequenos da máscara binarizada...")
    processed_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=500).astype(np.uint8)
    return processed_mask

def predict_3d(model, image_path, output_path, true_mask_path):
    """Realiza a predição da máscara 3D e a salva no caminho de saída."""
    try:
        print("Iniciando predição 3D...")
        
        # Carregando a imagem de entrada
        nifti_image = load_nifti_image(image_path)

        # Verificando se a imagem é um volume 3D
        if len(nifti_image.shape) != 3:
            raise ValueError("A imagem deve ser um volume 3D.")
        
        # Redimensionando a imagem para shape esperado pelo modelo (216, 256, 16)
        print("Redimensionando a imagem para shape (216, 256, 16)...")
        scale_factors = (216 / nifti_image.shape[0], 256 / nifti_image.shape[1], 32 / nifti_image.shape[2])
        resized_volume = zoom(nifti_image, scale_factors, order=1).astype('float32') / 255.0
        print(f"Imagem redimensionada para shape: {resized_volume.shape}")

        # Adicionando dimensões para predição
        resized_volume = np.expand_dims(resized_volume, axis=-1)  # Dimensão de canais
        resized_volume = np.expand_dims(resized_volume, axis=0)  # Dimensão de batch
        print(f"Shape do volume preparado para predição: {resized_volume.shape}")

        # Realizando a predição
        print("Realizando a predição da máscara...")
        prediction = model.predict(resized_volume)
        print("Predição concluída!")

        # Extraindo a máscara predita
        pred_mask = prediction[0, :, :, :, 0]  # Extraindo a máscara 3D
        print(f"Shape da máscara predita: {pred_mask.shape}")

        gt_img = nib.load(true_mask_path)
        #data_gt_img = np.array(true_mask_path.get_fdata())
        gt_header = gt_img.header
        gt_spacing = gt_header.get_zooms()
        gt_shape = gt_img.shape
        gt_physical_size = [gt_spacing[i] * gt_shape[i] for i in range(3)]

        value_size_x = gt_physical_size[0]  # Tamanho físico no eixo X
        value_size_y = gt_physical_size[1]  # Tamanho físico no eixo Y
        value_size_z = gt_physical_size[2]  # Tamanho físico no eixo Z

        # Definir os tamanhos físicos atuais e desejados nos eixos x, y e z
        current_spacing_x = 1.0  # Espaçamento atual em mm no eixo x (ajuste conforme necessário)
        current_spacing_y = 1.0  # Espaçamento atual em mm no eixo y (ajuste conforme necessário)
        current_spacing_z = 1.0  # Espaçamento atual em mm no eixo z (ajuste conforme necessário)

        target_physical_size_x = value_size_x  # Tamanho físico desejado em x (em mm)
        target_physical_size_y = value_size_y  # Tamanho físico desejado em y (em mm)
        target_physical_size_z = value_size_z   # Tamanho físico desejado em z (em mm)

        # Calculando os tamanhos físicos atuais
        current_physical_size_x = pred_mask.shape[0] * current_spacing_x
        current_physical_size_y = pred_mask.shape[1] * current_spacing_y
        current_physical_size_z = pred_mask.shape[2] * current_spacing_z

        # Calculando os fatores de redimensionamento para cada eixo
        resize_factor_x = target_physical_size_x / current_physical_size_x
        resize_factor_y = target_physical_size_y / current_physical_size_y
        resize_factor_z = target_physical_size_z / current_physical_size_z

        print(f"Fator de redimensionamento no eixo x: {resize_factor_x}")
        print(f"Fator de redimensionamento no eixo y: {resize_factor_y}")
        print(f"Fator de redimensionamento no eixo z: {resize_factor_z}")

        # Aplicar redimensionamento em x, y e z
        adjusted_pred_mask = zoom(pred_mask, (resize_factor_x, resize_factor_y, resize_factor_z), order=0)
        print(f"Shape da máscara ajustada: {adjusted_pred_mask.shape}")

        # Processando a máscara ajustada
        pred_mask_processed = postprocess_mask(adjusted_pred_mask, percentile=90)

        # Carregando a máscara verdadeira (golden truth)
        true_mask = load_nifti_image(true_mask_path)

        # Garantindo que a true_mask e a máscara predita tenham o mesmo shape
        if true_mask.shape != pred_mask_processed.shape:
            print("Redimensionando a máscara verdadeira para corresponder ao shape da máscara predita...")
            true_mask = zoom(true_mask, (
                pred_mask_processed.shape[0] / true_mask.shape[0],
                pred_mask_processed.shape[1] / true_mask.shape[1],
                pred_mask_processed.shape[2] / true_mask.shape[2]
            ), order=0)

        # Refinando a máscara predita para incluir apenas a ROI da golden truth
        refined_mask = pred_mask_processed * (true_mask > 0).astype(np.uint8)
        print("Máscara refinada com sucesso!")

        # Salvando a máscara processada
        save_nifti_image(output_path, refined_mask)


    except OSError as e:
        print(f"Erro de sistema operacional: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

