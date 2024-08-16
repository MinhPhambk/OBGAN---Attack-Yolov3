import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from OBGAN import OBGAN
from yolov3_infer import model

if __name__ == "__main__":
    use_cuda=True
    image_nc=3
    epochs = 600
    batch_size = 128
    C_TRESH =  1  #c cang thap thi nhieu cang min
    BOX_MIN = 0
    BOX_MAX = 1
    # increase to speed up training phase but be carefull of how model is gonna be converged
    # Define what device we are using
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("CUDA Available: ",torch.cuda.is_available())
    print("Using device: ", device)

    #Gene: input tam hinh goc  >> pert : encoder sau conv2d lam nho cai hinh down scaling >> up sampling  decode lam to cai hinh lai >> hinh goc
    #Dicri: input  Output G (pert) >> d_fake_probs output xac xuat D , D_fake : 1 tam hinh nha ra 1 phantu ( kiem tra pert co thay doi dc label ko)

    #MODEL_Taget
    MODEL_NAME = "YOLOv3"

    print("Successfully loaded target model ", MODEL_NAME)

    model_num_labels = 80
    CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
    
    model_num_labels = 80

    stop_epoch = 5    
            
    #coco_dataset = torchvision.datasets.ImageFolder('./dataset/test_tunning', transform=transforms.ToTensor())
    coco_dataset = torchvision.datasets.ImageFolder('./dataset/dog_cat_person_bear/train', transform=transforms.ToTensor())
    dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("training image examples: ", coco_dataset.__len__())

    obGAN = OBGAN(device,
                model,
                model_num_labels,
                image_nc,
                stop_epoch,
                BOX_MIN,
                BOX_MAX,
                C_TRESH,
                class_name = CLASS_NAMES,
                batsize = batch_size,
                dataset_name="MS_COCO",
                is_targeted=False)

    obGAN.train(dataloader, epochs)