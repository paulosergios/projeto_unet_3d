import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix

def dice_coefficient(y_true, y_pred):
    """
    Calcula o Dice Coefficient entre duas máscaras binárias.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    sum_masks = np.sum(y_true) + np.sum(y_pred)
    if sum_masks == 0:
        return 1.0  # Se ambas as máscaras estiverem vazias, consideramos perfeito.
    return 2.0 * intersection / sum_masks

# Caminhos dos arquivos NIfTI
pred_path = "C:/Users/paulo/Desktop/predict.nii.gz"
gt_path   = "C:/Users/paulo/Desktop/Dataset/database/testing/patient111/patient111_frame01_gt.nii.gz"

# Carregar as imagens NIfTI
pred_img = nib.load(pred_path)
gt_img   = nib.load(gt_path)

# Obter os dados (caso as imagens não estejam no mesmo shape, será necessário redimensionar)
pred_data = pred_img.get_fdata()
gt_data   = gt_img.get_fdata()

print("Shape da máscara predita:", pred_data.shape)
print("Shape da máscara ground truth original:", gt_data.shape)

# Se os shapes forem diferentes, redimensionar a ground truth para o shape da predição
if gt_data.shape != pred_data.shape:
    print("Redimensionando a ground truth para o shape da predição...")
    gt_data = resize(gt_data, pred_data.shape, order=0, preserve_range=True, anti_aliasing=False)
    print("Novo shape da ground truth:", gt_data.shape)

# Binarização das máscaras (ajuste o threshold conforme necessário)
pred_bin = (pred_data > 0.5).astype(np.uint8)
gt_bin   = (gt_data > 0.5).astype(np.uint8)

# Cálculo do Dice Coefficient
dice = dice_coefficient(gt_bin, pred_bin)
print("Dice Coefficient:", dice)

# Cálculo do Intersection over Union (IoU)
iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten())
print("Intersection over Union (IoU):", iou)

# Cálculo da acurácia
acc = accuracy_score(gt_bin.flatten(), pred_bin.flatten())
print("Accuracy:", acc)

# Calcular a matriz de confusão
cm = confusion_matrix(gt_bin.flatten(), pred_bin.flatten())
TN, FP, FN, TP = cm.ravel()

# Cálculos adicionais de métricas
total_pixels = cm.sum()
accuracy = (TP + TN) / total_pixels
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

FP_percent = (FP / total_pixels) * 100
FN_percent = (FN / total_pixels) * 100
TN_percent = (TN / total_pixels) * 100
TP_percent = (TP / total_pixels) * 100

print("Matriz de Confusão (detalhada):")
print(f"Verdadeiros Negativos (TN): {TN} ({TN_percent:.2f}%)")
print(f"Falsos Positivos (FP): {FP} ({FP_percent:.2f}%)")
print(f"Falsos Negativos (FN): {FN} ({FN_percent:.2f}%)")
print(f"Verdadeiros Positivos (TP): {TP} ({TP_percent:.2f}%)")

print("\nMétricas Calculadas:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensibilidade (Recall): {sensitivity:.4f}")
print(f"Especificidade: {specificity:.4f}")

print("\nAnalisando os True Positives:")
print(f"Total de True Positives: {TP}")
print(f"Área de interesse (ground truth) (1s): {np.sum(gt_bin)}")
print(f"Proporção de True Positives sobre a área de interesse: {TP / np.sum(gt_bin) * 100:.2f}%")

if TP > 0:
    print("O modelo está identificando áreas relevantes.")
else:
    print("O modelo não está identificando áreas relevantes.")

# Visualização com overlay (fatias)
def plot_overlay(slice_idx):
    plt.figure(figsize=(12, 5))
    
    # Máscara ground truth (azul) e predição (vermelho)
    overlay = np.zeros((*gt_bin.shape[:2], 3))
    overlay[..., 2] = gt_bin[:, :, slice_idx]  # Azul para ground truth
    overlay[..., 0] = pred_bin[:, :, slice_idx]  # Vermelho para predição
    
    plt.subplot(1, 2, 1)
    plt.imshow(overlay)
    plt.title(f'Overlay (Slice {slice_idx})')
    plt.axis('off')
    
    # Heatmap de diferenças
    diff = np.abs(gt_bin[:, :, slice_idx] - pred_bin[:, :, slice_idx])
    plt.subplot(1, 2, 2)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Heatmap de Diferenças (Slice {slice_idx})')
    plt.axis('off')
    
    plt.show()

# Plotando para uma fatia intermediária
plot_overlay(gt_bin.shape[2] // 2)
