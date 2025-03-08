#!/bin/bash
#SBATCH --job-name=treinamento_unet         # Nome do job
#SBATCH --partition=gpu-8-h100
#SBATCH --gres=gpu:4                      # Solicitar 4 GPU
#SBATCH --cpus-per-task=16                # Número de CPUs por tarefa
#SBATCH --mem=200G                          # Memória total disponível para o job
#SBATCH --time=02-00:00:00         

# Carregar módulos necessários
module load cuda/11.8                       # Substituir pela versão de CUDA disponível
module load tensorflow/2.12                 # Substituir pela versão de TensorFlow disponível

# Executar script
python /home/psdsjunior/meu_ambiente/projeto_unet/script/train.py
