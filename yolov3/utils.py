import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import warnings

# Tắt tất cả các cảnh báo từ PyTorch
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


def gpu2cpu_long(gpu_matrix):
    """ place float gpu tensor to long cpu tensor """
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def find_all_boxes(
    output,
    device,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False):
    """ extracting bboxes and confidece from output """
    num_classes, num_anchors = int(num_classes), int(num_anchors)
    anchor_step = int(len(anchors) / num_anchors)
    if output.dim == 3:
       output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)
   
    grid_x = torch.linspace(0, h-1, h).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    grid_y = torch.linspace(0, w-1, w).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
   
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
 
    det_confs = torch.sigmoid(output[4])
 
    cls_confs = nn.Softmax(dim=0)(output[5: 5 + num_classes].transpose(0, 1)).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
 
    sz_hw  = h * w
    sz_hwa = sz_hw * num_anchors
 
    det_confs     = det_confs.cpu()
    cls_max_confs = cls_max_confs.cpu()
    cls_max_ids   = gpu2cpu_long(cls_max_ids)
    xs, ys = xs.cpu(), ys.cpu()
    ws, hs = ws.cpu(), hs.cpu()
 
    if validation:
        cls_confs = cls_confs.view(-1, num_classes).cpu()
 
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
 
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw  = ws[ind]
                        bh  = hs[ind]
 
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id   = cls_max_ids  [ind]
                        box =[bcx/w, bcy/h, bw/608, bh/608, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def iou(box1, box2, x1y1x2y2 = True):
    """ Intersection Over Union """
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
 
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else: #(x, y, w, h)
        mx = min(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        Mx = max(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
        my = min(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
        My = max(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
 
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
 
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
   
    corea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    corea = cw * ch
    uarea = area1 + area2 - corea
    return corea / uarea


def nms(boxes, nms_thresh):
    """ None Max Separetion """
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]
 
    _, sortIds = torch.sort(det_confs)
    out_boxes =[]
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
           out_boxes.append(box_i)
           for j in range(i + 1, len(boxes)):
               box_j = boxes[sortIds[j]]
               if iou(box_i, box_j, x1y1x2y2 = False) > nms_thresh:
                  box_j[4] = 0
    return out_boxes


def filtered_boxes(model, device, img, conf_thresh, nms_thresh):
    """ filter best boxes from all boxes """
    model.eval()
   
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img).unsqueeze(0)
    elif type(img) == np.ndarray:
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print('unknown image type')
        exit(-1)
 
    img = img.to(device)
 
    output = model(img)
 
    boxes =[]
    for i in range(3):
        boxes += find_all_boxes(
            output[i].data,
            device,
            conf_thresh,
            model.num_classes,
            model.anchors[i],
            model.num_anchors)[0]
 
    boxes = nms(boxes, nms_thresh)
 
    return boxes

def filtered_boxes_im4(model, device, img, conf_thresh, nms_thresh):
    """ filter best boxes from all boxes """
    model.eval()
 
    img = img.to(device)
 
    output = model(img)

    boxes =[]
    for i in range(3):
        boxes += find_all_boxes(
            output[i].data,
            device,
            conf_thresh,
            model.num_classes,
            model.anchors[i],
            model.num_anchors)[0]
    
    boxes = nms(boxes, nms_thresh)
 
    return boxes

def iou_tensor(boxes1, boxes2, x1y1x2y2=True):
    """Intersection Over Union optimized with tensors."""
    if x1y1x2y2:
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    else:
        x1 = torch.max(boxes1[:, 0] - boxes1[:, 2] / 2, boxes2[:, 0] - boxes2[:, 2] / 2)
        y1 = torch.max(boxes1[:, 1] - boxes1[:, 3] / 2, boxes2[:, 1] - boxes2[:, 3] / 2)
        x2 = torch.min(boxes1[:, 0] + boxes1[:, 2] / 2, boxes2[:, 0] + boxes2[:, 2] / 2)
        y2 = torch.min(boxes1[:, 1] + boxes1[:, 3] / 2, boxes2[:, 1] + boxes2[:, 3] / 2)

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    if x1y1x2y2:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area
    return iou

def nms_tensor(boxes, nms_thresh):
    """Optimized Non-Maximum Suppression using tensors."""
    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = boxes[:, 4]
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        remaining_boxes = boxes[order[1:]]
        ious = iou_tensor(boxes[i].unsqueeze(0), remaining_boxes, x1y1x2y2=False)

        order = order[1:][ious <= nms_thresh]

    return boxes

def nms_combined(boxes, nms_thresh, x1y1x2y2=False):
    """Optimized NMS and IoU combined into a single function using tensors."""
    if len(boxes) == 0:
        return []

    # Convert list of boxes to tensor
    boxes = torch.stack(boxes)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = boxes[:, 4]
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        
        if order.numel() == 1:
            break

        remaining_boxes = boxes[order[1:]]
        
        if x1y1x2y2:
            x1 = torch.max(boxes[i, 0], remaining_boxes[:, 0])
            y1 = torch.max(boxes[i, 1], remaining_boxes[:, 1])
            x2 = torch.min(boxes[i, 2], remaining_boxes[:, 2])
            y2 = torch.min(boxes[i, 3], remaining_boxes[:, 3])
        else:
            x1 = torch.max(boxes[i, 0] - boxes[i, 2] / 2, remaining_boxes[:, 0] - remaining_boxes[:, 2] / 2)
            y1 = torch.max(boxes[i, 1] - boxes[i, 3] / 2, remaining_boxes[:, 1] - remaining_boxes[:, 3] / 2)
            x2 = torch.min(boxes[i, 0] + boxes[i, 2] / 2, remaining_boxes[:, 0] + remaining_boxes[:, 2] / 2)
            y2 = torch.min(boxes[i, 1] + boxes[i, 3] / 2, remaining_boxes[:, 1] + remaining_boxes[:, 3] / 2)
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        if x1y1x2y2:
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        else:
            area1 = boxes[i, 2] * boxes[i, 3]
            area2 = remaining_boxes[:, 2] * remaining_boxes[:, 3]
        
        union_area = area1 + area2 - inter_area
        ious = inter_area / union_area

        # Keep boxes with IoU less than the threshold
        order = order[1:][ious <= nms_thresh]

    return boxes

def find_all_boxes_optimized(
    output,
    device,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False):
    """Extract bounding boxes and confidence from output using optimized tensor operations."""
    num_classes = int(num_classes)
    num_anchors = int(num_anchors)
    anchor_step = len(anchors) // num_anchors
    
    if output.dim() == 3:
        output = output.unsqueeze(0)
        
    batch = output.size(0)
    h, w = output.size(2), output.size(3)
    
    # Reshape and transpose the output tensor
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)
    
    # Create grid for x and y
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(-1).to(device)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(-1).to(device)
    
    # Calculate x, y, w, h
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
    
    anchors = torch.tensor(anchors, dtype=torch.float32).view(num_anchors, anchor_step).to(device)
    anchor_w = anchors[:, 0].repeat(batch, 1).repeat(1, h * w).view(-1)
    anchor_h = anchors[:, 1].repeat(batch, 1).repeat(1, h * w).view(-1)
    
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    
    det_confs = torch.sigmoid(output[4])
    
    # Calculate class confidence and class id
    cls_confs = nn.Softmax(dim=0)(output[5: 5 + num_classes].transpose(0, 1))
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    
    # Convert to CPU for filtering
    det_confs, cls_max_confs, cls_max_ids = det_confs.cpu(), cls_max_confs.cpu(), cls_max_ids.cpu()
    xs, ys, ws, hs = xs.cpu(), ys.cpu(), ws.cpu(), hs.cpu()
    
    if validation:
        cls_confs = cls_confs.view(-1, num_classes).cpu()
    
    all_boxes = []
    
    for b in range(batch):
        boxes = []
        start_idx = b * num_anchors * h * w
        for i in range(num_anchors):
            ind = start_idx + i * h * w
            
            mask = det_confs[ind:ind + h * w] > conf_thresh
            if mask.any():
                bcx = xs[ind:ind + h * w][mask]
                bcy = ys[ind:ind + h * w][mask]
                bw = ws[ind:ind + h * w][mask]
                bh = hs[ind:ind + h * w][mask]
                det_conf = det_confs[ind:ind + h * w][mask]
                cls_max_conf = cls_max_confs[ind:ind + h * w][mask]
                cls_max_id = cls_max_ids[ind:ind + h * w][mask]
                
                # Construct boxes
                for j in range(bcx.size(0)):
                    box = [bcx[j] / w, bcy[j] / h, bw[j] / 608, bh[j] / 608, det_conf[j], cls_max_conf[j], cls_max_id[j]]
                    if not only_objectness and validation:
                        for c in range(num_classes):
                            tmp_conf = cls_confs[ind + j][c]
                            if c != cls_max_id[j] and det_conf[j] * tmp_conf > conf_thresh:
                                box.append(tmp_conf)
                                box.append(c)
                    boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes

def find_all_boxes_2(
    output,
    device,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False):
    """ Extracting bboxes and confidence from output """
    num_classes, num_anchors = int(num_classes), int(num_anchors)
    anchor_step = int(len(anchors) / num_anchors)
    
    if output.dim() == 3:
        output = output.unsqueeze(0)
    
    batch = output.size(0)
    assert(output.size(1) == (5 + num_classes) * num_anchors)
    h, w = output.size(2), output.size(3)
    
    all_boxes = []

    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)
    
    grid_x = torch.linspace(0, w-1, w, device=device).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w)
    grid_y = torch.linspace(0, h-1, h, device=device).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w)
    
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
    
    anchors = torch.tensor(anchors, device=device).view(num_anchors, anchor_step)
    anchor_w = anchors[:, 0]
    anchor_h = anchors[:, 1]
    
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    
    det_confs = torch.sigmoid(output[4])
    
    cls_confs = nn.Softmax(dim=0)(output[5: 5 + num_classes].transpose(0, 1))
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    
    if validation:
        cls_confs = cls_confs.view(-1, num_classes)

    for b in range(batch):
        start_idx = b * sz_hwa
        end_idx = start_idx + sz_hwa
        batch_det_confs = det_confs[start_idx:end_idx]
        batch_cls_max_confs = cls_max_confs[start_idx:end_idx]
        batch_cls_max_ids = cls_max_ids[start_idx:end_idx]
        batch_xs = xs[start_idx:end_idx]
        batch_ys = ys[start_idx:end_idx]
        batch_ws = ws[start_idx:end_idx]
        batch_hs = hs[start_idx:end_idx]
        
        if only_objectness:
            confs = batch_det_confs
        else:
            confs = batch_det_confs * batch_cls_max_confs
        
        mask = confs > conf_thresh
        filtered_xs = batch_xs[mask]
        filtered_ys = batch_ys[mask]
        filtered_ws = batch_ws[mask]
        filtered_hs = batch_hs[mask]
        filtered_confs = confs[mask]
        filtered_cls_max_confs = batch_cls_max_confs[mask]
        filtered_cls_max_ids = batch_cls_max_ids[mask]
        
        if validation:
            batch_cls_confs = cls_confs[start_idx:end_idx]
        
        boxes = torch.stack([
            filtered_xs / w,
            filtered_ys / h,
            filtered_ws / 608,
            filtered_hs / 608,
            filtered_confs,
            filtered_cls_max_confs,
            filtered_cls_max_ids.float()
        ], dim=1)
        
        if not only_objectness and validation:
            for c in range(num_classes):
                tmp_conf = batch_cls_confs[mask][:, c]
                tmp_mask = filtered_confs * tmp_conf > conf_thresh
                tmp_boxes = boxes[tmp_mask]
                if tmp_boxes.size(0) > 0:
                    boxes = torch.cat([boxes, tmp_boxes], dim=0)
        
        all_boxes.append(boxes)

    return all_boxes