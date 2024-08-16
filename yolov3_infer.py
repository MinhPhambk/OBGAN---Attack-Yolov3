from yolov3 import cfg
from yolov3.utils import *
import numpy as np
import torchvision
import torch.nn.functional as F
import cv2
import torch

# class IdentityLinear(nn.Module):
#     def __init__(self, input_size=80):
#         super(IdentityLinear, self).__init__()
#         self.linear = nn.Linear(input_size, input_size, bias=True)  # Input size to output size

#         # Initialize weight matrix as identity and bias vector as zeros
#         self.linear.weight.data = torch.eye(input_size, dtype=torch.float64)
#         self.linear.bias.data = torch.zeros(input_size, dtype=torch.float64)

#     def forward(self, x):
#         return self.linear(x)

ROOT = './yolov3'  # YOLOv3 root directory
# convert_gradient = IdentityLinear()


def model(dataset=None, conf_thres=0.005, device=None, one_img=False, label_mode=False):
    if device == None: device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    names = cfg.CLASS_NAMES

    # Load model
    model = cfg.MODEL
    model.eval()
    nms_thresh = 0.5
    conf_thres = 0.0005

    if one_img:
        batch = 1
        im = dataset
        im = F.interpolate(im.unsqueeze(0), size=(608, 608), mode='bilinear', align_corners=False)
        output = model(im)
        boxes = []
        for i in range(3):
            boxes += find_all_boxes_2(
                output[i].data,
                device,
                conf_thres,
                model.num_classes,
                model.anchors[i],
                model.num_anchors)[0]
        boxes = nms_combined(boxes, nms_thresh)
        res = torch.zeros((80), device=device)
        conf = boxes[:, -2]
        cls = boxes[:, -1].long()
        for c in range(80):
            mask = (cls == c)
            if mask.any():
                res[c] = conf[mask].max()
        return torch.argmax(res).item()

    else:
        if isinstance(dataset, str):        
            # Run inference
            batch = 1

            im = cv2.imread(dataset)
            im = torch.from_numpy(im.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            im = im.to(device)
            output = model(im)
            boxes = []
            for i in range(3):
                boxes += find_all_boxes_2(
                    output[i].data,
                    device,
                    conf_thres,
                    model.num_classes,
                    model.anchors[i],
                    model.num_anchors)[0]
            if len(boxes) == 0:
                for i in range(3):
                    boxes += find_all_boxes_2(
                        output[i].data,
                        device,
                        conf_thres/50,
                        model.num_classes,
                        model.anchors[i],
                        model.num_anchors)[0]
            if len(boxes) == 0:
                for i in range(3):
                    boxes += find_all_boxes_2(
                        output[i].data,
                        device,
                        conf_thres/500,
                        model.num_classes,
                        model.anchors[i],
                        model.num_anchors)[0]
            boxes = nms_combined(boxes, nms_thresh) 
            if len(boxes) == 0: return torch.randint(0, 80, (1,)).item()
            res = torch.zeros((80), device=device)
            conf = boxes[:, -2]
            cls = boxes[:, -1].long()
            for c in range(80):
                mask = (cls == c)
                if mask.any():
                    res[c] = conf[mask].max()
            return torch.argmax(res).item()
        
        else:
            # Run inference
            batch = dataset.shape[0]
            res = torch.zeros(batch, 80).to(device)
            
            for j, im in enumerate(dataset):
                im = F.interpolate(im.unsqueeze(0), size=(608, 608), mode='bilinear', align_corners=False)
                output = model(im)
                boxes = []
                for i in range(3):
                    boxes += find_all_boxes_2(
                        output[i].data,
                        device,
                        conf_thres,
                        model.num_classes,
                        model.anchors[i],
                        model.num_anchors)[0]
                boxes = nms_combined(boxes, nms_thresh)
                conf = boxes[:, -2]
                cls = boxes[:, -1].long()
                for c in range(80):
                    mask = (cls == c)
                    if mask.any():
                        res[j][c] = conf[mask].max()

            return res