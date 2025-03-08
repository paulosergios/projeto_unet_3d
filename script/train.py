import numpy as np
import os
import tensorflow as tf
import nibabel as nib
import gc
import psutil
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet_model_3d import build_unet_3d as build_unet
from transformer_3d import build_transformer_3d as build_transformer
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from skimage.transform import resize, rotate
from tensorflow.keras.optimizers import Adam
from skimage.measure import regionprops
from skimage.util import random_noise
import random

# Configurações de ambiente e GPU
os.environ["TF_DISABLE_MKL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gc.collect()

MODEL_TYPE = 'unet'

# Caminho para os dados
base_data_dir = '/home/psdsjunior/meu_ambiente/database/training'


def crop_to_roi(image, mask):
    """
    Recorta a imagem e a máscara para a menor caixa delimitadora que contenha a ROI.
    """
    regions = regionprops((mask > 0).astype(int))
    if len(regions) == 0:
        raise ValueError("Nenhuma ROI encontrada na máscara fornecida.")
    
    # Obtem os limites da ROI (caixa delimitadora em 3D)
    min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox
    
    # Recorta a imagem e a máscara
    cropped_image = image[min_row:max_row, min_col:max_col, min_depth:max_depth]
    cropped_mask = mask[min_row:max_row, min_col:max_col, min_depth:max_depth]
    
    return cropped_image, cropped_mask


def resize_with_padding(image, target_shape):
    """
    Redimensiona uma imagem para o tamanho alvo, mantendo a proporção e preenchendo com zeros.
    """
    target_h, target_w, target_d = target_shape
    current_h, current_w, current_d = image.shape

    scale = min(target_h / current_h, target_w / current_w, target_d / current_d)
    new_shape = (int(current_h * scale), int(current_w * scale), int(current_d * scale))

    resized = resize(image, new_shape, mode='constant', preserve_range=True)
    
    # Cria um novo array preenchido com zeros e centraliza a imagem redimensionada
    padded = np.zeros(target_shape, dtype=resized.dtype)
    start_h = (target_h - new_shape[0]) // 2
    start_w = (target_w - new_shape[1]) // 2
    start_d = (target_d - new_shape[2]) // 2
    padded[start_h:start_h + new_shape[0], start_w:start_w + new_shape[1], start_d:start_d + new_shape[2]] = resized
    
    return padded


def augment_data_3d(image, mask):
    """
    Aplica augmentação de dados 3D nas imagens e máscaras.
    """
    # Rotação aleatória
    angle = random.uniform(-10, 10)  # Rotação em graus
    image = rotate(image, angle, axes=(0, 1), mode='constant', preserve_range=True)
    mask = rotate(mask, angle, axes=(0, 1), mode='constant', preserve_range=True)

    # Flip aleatório
    if random.random() > 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    # Ruído aleatório
    image = random_noise(image, mode='gaussian', var=0.001)

    return image, mask


def load_patient_data(base_data_dir):
    """
    Carrega os dados dos pacientes e aplica processamento inicial.
    """
    print("Iniciando o carregamento dos dados dos pacientes...")
    images = []
    masks = []
    for i in range(1, 101):
        patient_id = f"patient{i:03d}"
        image_path = os.path.join(base_data_dir, patient_id, f"{patient_id}_frame01.nii.gz")
        mask_path = os.path.join(base_data_dir, patient_id, f"{patient_id}_frame01_gt.nii.gz")

        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = nib.load(image_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()

            if image.shape != mask.shape:
                print(f"Inconsistência de dimensões detectada para {patient_id}. Ignorando...")
                continue

            # Recorta a ROI
            cropped_image, cropped_mask = crop_to_roi(image, mask)
            
            # Redimensiona com padding para manter tamanho fixo
            target_shape = (216, 256, 32)
            image_resized = resize_with_padding(cropped_image, target_shape)
            mask_resized = resize_with_padding(cropped_mask, target_shape)

            images.append(np.expand_dims(image_resized, axis=-1))
            masks.append(np.expand_dims(mask_resized, axis=-1))

    print(f"Total de imagens carregadas: {len(images)}")
    return np.array(images, dtype='float32'), np.array(masks, dtype='float32')


def load_patient_data_with_split(base_data_dir, test_size=0.2):
    images, masks = load_patient_data(base_data_dir)
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=test_size, random_state=42)
    return train_images, val_images, train_masks, val_masks


class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mem = psutil.virtual_memory()
        print(f"Memória usada no final da epoch {epoch + 1}: {mem.used / (1024 ** 3):.2f} GB")


def data_generator(images, masks, batch_size):
    """
    Gerador de dados que aplica augmentação em tempo real.
    """
    while True:
        idxs = np.random.permutation(len(images))
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_images = []
            batch_masks = []
            for idx in batch_idxs:
                img, msk = augment_data_3d(images[idx], masks[idx])
                batch_images.append(img)
                batch_masks.append(msk)
            yield np.array(batch_images), np.array(batch_masks)


def train():
    print("Iniciando treinamento...")
    train_images, val_images, train_masks, val_masks = load_patient_data_with_split(base_data_dir)
    
    # Normaliza as imagens e binariza as máscaras
    print("Normalizando dados e processando máscaras...")
    train_images /= 255.0
    val_images /= 255.0
    train_masks = (train_masks > 0).astype('float32')
    val_masks = (val_masks > 0).astype('float32')

    input_shape = train_images.shape[1:]
    if MODEL_TYPE == 'unet':
        model = build_unet(input_size=input_shape)
        checkpoint_path = 'meu_ambiente/projeto_unet/models/unet/best_unet_model.keras'
    else:
        model = build_transformer(input_size=input_shape)
        checkpoint_path = 'meu_ambiente/projeto_unet/models/transformer/best_transformer_model.keras'

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    callbacks = [checkpoint, early_stopping, MemoryCallback()]

    batch_size = 4
    train_gen = data_generator(train_images, train_masks, batch_size)

    model.fit(
        train_gen,
        steps_per_epoch=len(train_images) // batch_size,
        validation_data=(val_images, val_masks),
        epochs=50,
        callbacks=callbacks
    )
    print(f"Modelo salvo em: {checkpoint_path}")


if __name__ == "__main__":
    train()
