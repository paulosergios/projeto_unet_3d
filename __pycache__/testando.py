import nibabel as nib
import matplotlib.pyplot as plt

# Caminho atualizado para a imagem
image_path = "C:/Users/paulo/Desktop/patient101_4d.nii/patient101_4d.nii"

    # Carregar a imagem
img = nib.load(image_path)
data = img.get_fdata()

# Exibir a forma dos dados
print(f"Forma dos dados: {data.shape}")

    # Certificar-se de que o número de dimensões está correto
        # Exibir uma fatia da imagem (exemplo: primeira fatia do primeiro volume)
plt.imshow(data[:, :, 0, 1], cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(data[:, :, 0, 5], cmap='gray')
plt.axis('off')
plt.show()