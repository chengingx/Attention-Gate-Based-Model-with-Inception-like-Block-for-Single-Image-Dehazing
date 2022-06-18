import torch

# path = "dataset/RESIDE_ITS"
path = "dataset/OTS_BETA"
Hazy = path + "/hazy"
Clear = path + "/clear"
SAVE_FOLDER = "evaluation"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED = False
INITIAL_EPOCH = 0
PRETRAINED_EPOCH = 0
INITIAL_LEARNING_RATE = 1e-4
BATCH_SIZE = 5
IMAGE_SIZE = 256
LAMBDA1 = 0.001
LAMBDA2 = 1
NUM_EPOCHS = 400
SAVE_SAMPLE = False
RECORD_DATASET_INDICES = True