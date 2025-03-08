# Cardiac Image Segmentation Using U-Net

Welcome to the repository for my **Cardiac Image Segmentation** project. This repository contains the code and resources for a study conducted as part of my Master’s degree in **Systems and Computing**. The goal of this project is to accurately segment cardiac structures from 3D medical imaging using deep learning techniques.

## Dataset

The dataset used for this project is sourced from the **Human Heart Project** and can be accessed at the following link:  
[Human Heart Dataset](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc).

### Dataset Description
- **Type**: 3D medical imaging (MRI)
- **Size**: Contains 100 training samples and 50 test samples
- **Structure**: Each patient has:
  - Two MRI scans  
  - Two corresponding masks for segmentation

The dataset is an excellent resource for exploring deep learning applications in cardiac health.

## Objectives

The primary objectives of this project include:
1. **Developing a robust U-Net architecture**: Fine-tuned to work on 3D medical imaging.
2. **Improving segmentation quality**: By leveraging techniques such as pre-processing (denoising, normalization) and post-processing.
3. **Automating cardiac segmentation**: To provide accurate masks for medical analysis.

## Workflow

1. **Data Preprocessing**:
   - Images are normalized and optionally denoised using wavelet techniques.
   - Masks are processed to match the input dimensions.
   
2. **Model Training**:
   - The U-Net architecture is trained using the provided dataset.
   - Training and validation checkpoints are saved for optimal model selection.

3. **Prediction**:
   - Segmentation results are output as NIfTI files for 3D visualization and analysis.
   - Metrics such as Dice Similarity Coefficient (DSC) and Jaccard Index are used for evaluation.

4. **Visualization**:
   - Segmentation results are visualized to compare predicted and ground-truth masks.

## Repository Structure

```
├── data/
│   ├── raw/               # Raw dataset files
│   ├── processed/         # Processed images and masks
├── models/
│   ├── unet_model.py      # U-Net architecture implementation
│   ├── train.py           # Training script with checkpointing
│   ├── predict_3d.py      # 3D prediction script
├── results/
│   ├── masks/             # Predicted masks
│   ├── metrics/           # Evaluation metrics
├── README.md              # Project documentation
```

## Requirements

To run the code, ensure you have the following dependencies installed:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Nibabel
- Scikit-image

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
Run the following command to train the model:
```bash
python train.py
```

### Predicting and Visualizing
To perform predictions on new images:
```bash
python predict_3d.py
```

The predicted masks will be saved in the `results/masks/` folder.

## Acknowledgments

This project is part of my Master’s thesis, and I would like to thank the **Human Heart Project** for providing the dataset and my academic mentors for their guidance.

## Contact

For any questions or collaborations, feel free to contact me via email: [paulosergio@ufrn.edu.br].
