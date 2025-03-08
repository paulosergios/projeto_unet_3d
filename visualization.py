import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Função para ajustar a máscara verdadeira para a forma da máscara predita
def adjust_mask(true_mask, predicted_mask):
    # Verificar se as máscaras têm dimensões diferentes
    if true_mask.shape != predicted_mask.shape:
        # Calcular os fatores de redimensionamento
        factors = np.array(predicted_mask.shape) / np.array(true_mask.shape)
        print(f"Fatores de redimensionamento: {factors}")  # Debug

        # Ajustar a máscara verdadeira para o tamanho da máscara predita
        true_mask = zoom(true_mask, factors, order=1)
        print(f"Nova forma da máscara verdadeira após ajuste: {true_mask.shape}")  # Debug
    else:
        print("As máscaras já têm as mesmas dimensões, nenhum ajuste necessário.")
    return true_mask

def visualize_results(original, output_path, true_mask=None):
    # Carregar a máscara predita
    if isinstance(output_path, str):
        try:
            predicted = nib.load(output_path).get_fdata()
        except Exception as e:
            print(f"Erro ao carregar a máscara predita: {e}")
            return
    else:
        print("Caminho inválido para a máscara predita.")
        return

    plt.figure(figsize=(12, 6))

    # Verificar e exibir a imagem original
    if original is None:
        print("Imagem original não fornecida.")
        return
    if len(original.shape) == 3:
        original_slice = original[:, :, original.shape[2] // 2]
    else:
        original_slice = original  # Para 2D

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_slice, cmap='gray')
    plt.axis('off')

    # Ajustar a máscara verdadeira para as dimensões da predita, se necessário
    if true_mask is not None:
        true_mask = adjust_mask(true_mask, predicted)

    # Exibir a máscara predita
    if predicted is None:
        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask (None)")
        plt.text(0.5, 0.5, "No Prediction", horizontalalignment='center', verticalalignment='center')
    else:
        if len(predicted.shape) == 3:
            predicted_slice = predicted[:, :, predicted.shape[2] // 2]
        else:
            predicted_slice = predicted

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(predicted_slice.astype(np.float32), cmap='jet' , alpha=1)
        plt.axis('off')

    # Exibir a máscara verdadeira
    plt.subplot(1, 3, 3)
    plt.title("True Mask")
    if true_mask is None:
        plt.text(0.5, 0.5, "No Mask", horizontalalignment='center', verticalalignment='center')
    else:
        if len(true_mask.shape) == 3:
            true_mask_slice = true_mask[:, :, true_mask.shape[2] // 2]
        else:
            true_mask_slice = true_mask
        plt.imshow(true_mask_slice.astype(np.float32), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
