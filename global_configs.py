import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 47
TEXT_DIM = 50

# MOSEI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768

# python multimodal_driver.py --dataset mosei
