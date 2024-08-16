import torch
from yolov3.model import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = load_model('./yolov3/weights/yolov3.weights', DEVICE)
WEIGHTS = './yolov3/weights/yolov3.weights'
with open('./yolov3/class_names', 'r') as f:
    CLASS_NAMES = f.read().split('\n')