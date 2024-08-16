import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Gan_model
from yolov3_infer import model as target_model
from coco_models import ResnetGenerator as Generator
import cv2
import numpy as np
import time
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from skimage import io
import math
import os

dog_paths = []
for root, dirs, files in os.walk("C:/Users/phiho/Projects/Dataset/MS-COCO/COCO_81/coco81/train/dog"):
    for file in files:
        dog_paths.append(os.path.join(root, file).split("\\")[-1])

cat_paths = []
for root, dirs, files in os.walk("C:/Users/phiho/Projects/Dataset/MS-COCO/COCO_81/coco81/train/cat"):
    for file in files:
        cat_paths.append(os.path.join(root, file).split("\\")[-1])

person_paths = []
for root, dirs, files in os.walk("C:/Users/phiho/Projects/Dataset/MS-COCO/COCO_81/coco81/train/person"):
    for file in files:
        person_paths.append(os.path.join(root, file).split("\\")[-1])


if __name__ == "__main__":
    start_time = time.time()
    print(start_time,'running')
    use_cuda=True
    image_nc=3
    batch_size = 1

    gen_input_nc = image_nc

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda")

    # load the pretrained model
    print('=================== yolov3.pt model ===============================================')
    # load the generator of adversarial examples
    pretrained_generator_path = 'C:/Users/phiho/Projects/GAN_Yolov3/models/netG.100'
    print('./models/netG.100')
    pretrained_G = Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    for kcount in range(1,11):
        start_time_gen = time.time()
        print(start_time_gen,' running gen ',kcount)
   
    # test adversarial examples in cifar10 testing dataset
        coco_dataset = torchvision.datasets.ImageFolder('./dataset/test_2000_3class', transform=transforms.ToTensor())
        test_dataloader = DataLoader(coco_dataset, batch_size=1, shuffle=False, drop_last=True)

        print("len(test_dataloader): ", len(test_dataloader))
        num_correct = 0
        num_all = 0
        spsnr = 0
        sssim = 0
        count = 0
        sl0 = 0.0
        sl1 = 0.0
        sl2 = 0.0
        CLASS_NAMES = [ "cat", "dog", "person"]
        CLASS_NAMES_DICT = {"cat": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                            "dog": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                            "person": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0}
                        }
    
        print(kcount,'*perturbation')
        for i, data in enumerate(test_dataloader, 0):
            test_img, test_label = data
            name, _ = test_dataloader.dataset.samples[i]
            
            # test_label
            name = name.split("\\")[-1]
            if name in person_paths:
                test_label = 0
            elif name in dog_paths:
                test_label = 16
            elif name in cat_paths:
                test_label = 17

            test_img, test_label = test_img.to(device), test_label.to(device)

            targeted_rslt = target_model(test_img, device=device)
            targeted_lb = torch.argmax(targeted_rslt, 1)
            print(targeted_lb)
            break
            if (torch.sum(test_label==targeted_lb, 0) == 0):
                continue
            targeted_score = torch.nn.Softmax()(targeted_rslt[0])[targeted_lb[0]].detach().cpu().numpy()

            perturbation = pretrained_G(test_img)
            perturbation = torch.clamp(perturbation, -0.3, 0.3)
            # thay doi mean thhanh mediant >>
            perturbation_mean = perturbation.mean(dim=[1,2,3])
            perturbation[0] = perturbation[0] - perturbation_mean[0]

            #adv_img = test_img + k(tu 0.1 den 1.0) *perturbation
            adv_img = test_img + int(kcount)*0.1*perturbation

            adv_img = torch.clamp(adv_img, 0, 1)

            pred_rslt = target_model(adv_img)
            pred_lab = torch.argmax(pred_rslt, 1)
            pred_scores = torch.nn.Softmax()(pred_rslt[0])[pred_lab[0]].detach().cpu().numpy()

            num_all += len(test_label)
            current_correct = torch.sum(pred_lab==test_label,0)
            num_correct += current_correct

            # quick test for batch size = 1
            CLASS_NAMES_DICT[CLASS_NAMES[test_label[0].detach().cpu().numpy()]]["gt"] += 1
            if (current_correct.item() != 0):
                continue
            CLASS_NAMES_DICT[CLASS_NAMES[test_label[0].detach().cpu().numpy()]]["adv_succeed"] += 1
            if (current_correct.item() != 0):
                continue

            count += 1
            
            cv2.imwrite(f"./preview/0_{kcount}/id_{count}-0origin -label_{test_label[0].detach().cpu().numpy()}-classname_{CLASS_NAMES[test_label[0].detach().cpu().numpy()]}.jpg",
                                test_img[0].permute((1,2,0)).detach().cpu().numpy()*255)

            cv2.imwrite(f"./preview/0_{kcount}/id_{count}-1targerted-label_{targeted_lb[0].detach().cpu().numpy()}_{targeted_score}-classname_{CLASS_NAMES[targeted_lb[0].detach().cpu().numpy()]}.jpg",
                                test_img[0].permute((1,2,0)).detach().cpu().numpy()*255)

            cv2.imwrite(f"./preview/0_{kcount}/id_{count}-2pert.jpg",
                                perturbation[0].permute((1,2,0)).detach().cpu().numpy()*255)

            cv2.imwrite(f"./preview/0_{kcount}/id_{count}-3adv-label_{pred_lab[0].detach().cpu().numpy()}_{pred_scores}-classname_{CLASS_NAMES[pred_lab[0].detach().cpu().numpy()]}.jpg",
                                adv_img[0].permute((1,2,0)).detach().cpu().numpy()*255)
            image1 = io.imread(f"./preview/0_{kcount}/id_{count}-0origin -label_{test_label[0].detach().cpu().numpy()}-classname_{CLASS_NAMES[test_label[0].detach().cpu().numpy()]}.jpg")
            image2 = io.imread(f"./preview/0_{kcount}/id_{count}-3adv-label_{pred_lab[0].detach().cpu().numpy()}_{pred_scores}-classname_{CLASS_NAMES[pred_lab[0].detach().cpu().numpy()]}.jpg")
# Chuyển hình ảnh thành tensor PyTorch
            image1_tensor = to_tensor(image1).unsqueeze(0)
            image2_tensor = to_tensor(image2).unsqueeze(0)

# Chuyển sang kiểu dữ liệu float32 và chuẩn hóa giá trị về khoảng [0, 1]
            image1_tensor = image1_tensor.type(torch.float32) / 255.0
            image2_tensor = image2_tensor.type(torch.float32) / 255.0

# Tính chỉ số SSIM sử dụng hàm tích hợp trong PyTorch
            ssim_score = float(1 - F.mse_loss(image1_tensor, image2_tensor).cpu().numpy())
            l1 = float(F.l1_loss(image1_tensor,image2_tensor))
            l0 = float(torch.norm(abs(image1_tensor - image2_tensor)))
            print(f"SSIM score: {ssim_score}","   id  ",count)
            
            mse = F.mse_loss(image1_tensor, image2_tensor)
            psnr = 10 * torch.log10(1.0 / torch.sqrt(mse))
            print(f"PSNR: {psnr.item()}","   id  ",count)
            if(math.isinf(psnr.item())==False):
                spsnr += psnr.item()
            print("Sum SPSNR: ", spsnr)
            # tinh L0 L1 L2 LPnorm
            # Tính l2
            l2 = float(mse.cpu().numpy())

            print(f"L0 score: {l0:.21f}","   id  ",count)
            print(f"L1 score: {l1:.21f}","   id  ",count)
            print(f"L2 score: {l2:.21f}","   id  ",count)

            sl0 += l0
            sl1 += l1
            sl2 += l2
       
            print("SUM ", count," L0  : ", f"{sl0:.21f}"," -L1 : ", f"{sl1:.21f}"," -L2 : ", f"{sl2:.21f}" )
            print("Sum SPSNR: ", spsnr)
            sssim += ssim_score
            print("Sum SSSIM: ", sssim)
        print('SSIM: TB ',sssim/count) 
        print('PSNR: TB ',spsnr/count) 
        print("AVG L0  : ", f"{sl0/count:.21f}"," -L1 : ", f"{sl1/count:.21f}"," -L2 : ", f"{sl2/count:.21f}" )    
        print('cifar10 test dataset:')
        print('num_examples: ', num_all)
        print('num_correct: ', num_correct.item())
        print('adv_correct: ', count)
        print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/num_all))
        print('adv succeeded %f\n'%(1-num_correct.item()/num_all))

        for cls_name in CLASS_NAMES:
            CLASS_NAMES_DICT[cls_name]["adv_succeed_%"] = CLASS_NAMES_DICT[cls_name]["adv_succeed"] / CLASS_NAMES_DICT[cls_name]["gt"]

        import json
        CLASS_NAMES_DICT = json.dumps(CLASS_NAMES_DICT, indent=4)
        print(CLASS_NAMES_DICT)
        print(time.time()-start_time_gen)
        print('==================================================================')
    print(time.time()-start_time)