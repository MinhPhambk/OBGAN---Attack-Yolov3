import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os
import cv2
from tqdm.auto import tqdm
import time
import shutil
# from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
import gc
# from advertorch.attacks.iterative_projected_gradient import LinfPGDAttack

# from confidence-calibrated-adversarial-training
def find_last_checkpoint(model_file):
    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name,'').replace('.', '') for i in range(len(state_files))] 
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = np.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adv_loss(probs_model, onehot_labels, is_targeted):   
    # C&W loss function
    real = torch.sum(onehot_labels * probs_model, dim=1)
    other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1) # loss adv giam thi asr tang 1. index tunning chỉ số càng nhỏ full rate càng tăng
    zeros = torch.zeros_like(other)
    if is_targeted:
        loss_adv = torch.sum(torch.max(other - real, zeros)) 
    else:
        loss_adv = torch.sum(torch.max(real - other, zeros)) 
    return loss_adv

class OBGAN:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 epoch_of_change,
                 box_min,
                 box_max,
                 c_tresh,
                 class_name,
                 batsize,
                 dataset_name,
                 is_targeted):
        
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.c_treshold = c_tresh 
        self.class_name = class_name
        self.batsize = batsize
        self.dataset_name = dataset_name
        self.is_targeted = is_targeted
        
        self.models_path = './models/'
        self.writer = SummaryWriter('./checkpoints/logs/', max_queue=100)

        self.gen_input_nc = image_nc

        self.epoch_of_change = epoch_of_change
  
        # self.attacker = LinfPGDAttack(self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #      nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=box_min, clip_max=box_max,
        #      targeted=self.is_targeted)

        if dataset_name=="MS_COCO":
            from Gan_model import Generator 
            from Gan_model import Discriminator 
        else:
            raise NotImplementedError('dataset [%s] is not implemented' % dataset_name)

        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)
        self.netG_file_name = self.models_path + 'netG'
        self.netDisc_file_name = self.models_path + 'netD'

        os.makedirs(self.models_path, exist_ok=True)

        # initialize all weights
        last_netG = find_last_checkpoint(self.netG_file_name)
        last_netDisc = find_last_checkpoint(self.netDisc_file_name)
        if last_netG is not None:
            self.netG.load_state_dict(torch.load(last_netG))
            self.netDisc.load_state_dict(torch.load(last_netDisc))
            *_, self.start_epoch = last_netG.split('.')
            self.iteration = None
            self.start_epoch = int(self.start_epoch)+1
        else:
            self.netG.apply(weights_init)
            self.netDisc.apply(weights_init)
            self.start_epoch = 1
            self.iteration = 0

       # initialize optimizers
        if self.dataset_name == "MS_COCO":
            lr = 10**(-7) #0.0000001 2. index tunning
        else:
            raise NotImplementedError('dataset [%s] is not implemented' % dataset_name)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=lr)
        #luu cai cai optimizer de lan sau train tiep                                     
        self.optG_file_name = self.models_path + 'optG'
        self.optD_file_name = self.models_path + 'optD'

        last_optG = find_last_checkpoint(self.optG_file_name)
        last_optD = find_last_checkpoint(self.optD_file_name)
        if last_optG is not None:
            self.optimizer_G.load_state_dict(torch.load(last_optG))
            self.optimizer_D.load_state_dict((torch.load(last_optD)))

        self._use_attacker = (self.start_epoch < self.epoch_of_change)



    def train_batch(self, x, labels):    
        # optimize D
        for _ in range(1):
            # # add a clipping trick
            perturbation = torch.clamp(self.netG(x), -self.c_treshold, self.c_treshold)
            adv_images = 2*perturbation + x 
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            adv_prob = self.model(adv_images).detach().cpu().numpy()
            adv_labels = []
            # print('G')
            # summary(self.netG, (3, 299, 299))
            # print('D')
            # summary(self.netDisc, (3, 299, 299))
            for i in range(len(adv_prob)):
                adv_labels.append(np.argmax(adv_prob[i]))

            # print('adv_labels ', adv_labels)
            label_count = 0
            fool_count = 0
            for i in range(len(labels)):
                label_count += 1
                if labels[i] != adv_labels[i]:
                    fool_count += 1
            fool_rate = fool_count / label_count * 100
            print(f"fool rate: {fool_rate:.2f}%")
            with open("./models/fool_rate_log.txt", 'a', encoding='utf-8') as file:
                file.write(str(fool_rate) + '\n')

            target_folder1 ="./data_temp/adv_temp"
            os.makedirs(target_folder1, exist_ok=True)
            for j in range(0,self.batsize):
                cv2.imwrite(f"./data_temp/adv_temp/perturbation_{j}.jpg", 
                            cv2.cvtColor(perturbation[j].permute((1,2,0)).detach().cpu().numpy()*255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"./data_temp/adv_temp/adv_images_{j}.jpg", 
                            cv2.cvtColor(adv_images[j].permute((1,2,0)).detach().cpu().numpy()*255, cv2.COLOR_RGB2BGR))
            self.optimizer_D.zero_grad()

            #if self._use_attacker:
                #pgd_images = self.attacker.perturb(x,labels) 
                #d_real_logits, d_real_probs = self.netDisc(pgd_images)
            #else:
            d_real_logits, d_real_probs = self.netDisc(x) 
            d_fake_logits, d_fake_probs = self.netDisc(adv_images.detach())
            # adv_images 1 hoac nhieu tam hinh tuy vao bathsize
            # vi du bathsize = 3 thi d_fake_probs cho ra 3 phan tu nhãn sẽ là nhãn 0 >> label goc thi nhan 1
            # generate labels for discriminator (optionally smooth labels for stability)
         
            smooth = 0.1
            d_labels_real = torch.ones_like(d_real_probs, device=self.device) * (1 - smooth) # 3 so 1
            d_labels_fake = torch.zeros_like(d_fake_probs, device=self.device) # 3 so 0
            
            # discriminator loss
            loss_D_real = F.mse_loss(d_real_probs, d_labels_real)
            loss_D_fake = F.mse_loss(d_fake_probs, d_labels_fake)
            loss_D_GAN = (6*loss_D_fake + 4*loss_D_real)   #loss D giam thi SSIM va PSNR tang 3. 
            loss_D_GAN.backward()
            self.optimizer_D.step()
        #print("Final D")
        gc.collect()

        # optimize G
        for _ in range(1):

            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            d_fake_logits,d_fake_probs = self.netDisc(adv_images.detach()) 
            loss_G_fake = F.mse_loss(d_fake_probs, torch.ones_like(d_fake_probs, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # # calculate perturbation norm / lam cai loss cang nho cang tot 
            loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            # co gang thay doi c_treshold sao cho loss_perturb - self.c_treshold <=0 tang
            loss_perturb = torch.max(loss_perturb - self.c_treshold, torch.zeros(1, device=self.device))
            loss_perturb = torch.mean(loss_perturb)

            # Lay ket qua tu model goc ra
            f_fake_logits = self.model(adv_images)  
      
            # Doi ket qua thanh xac suat
            f_fake_probs = F.softmax(f_fake_logits, dim=1)

            # if training is targeted, indicate how many examples classified as targets
            # else show accuraccy on adversarial images

            fake_accuracy = torch.mean((torch.argmax(f_fake_probs, 1) == labels).float())

            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels.long()]
            # Tinh loss giua adv va model goc cang cao cang tot
            loss_adv = adv_loss(f_fake_probs, onehot_labels, self.is_targeted)
            
            # feel freeze to change
            if self.dataset_name == "MS_COCO": #4. index tunning
                alambda = 10 #0.5 # 10
                alpha = 1 # 1
                beta = 0.5 # 0.5 #tăng b foolrate tang, loss G tang  
                #print("alambda: %.1f,  alpha : %.1f, beta: %.1f, \n" %
                  #(alambda, alpha, beta ))
            else:
                raise NotImplementedError('dataset [%s] is not implemented' % self.dataset_name)
            # tun 3 gia tri alpha alamda beta, gia tri cang cao the hien quan trong cua loss
            loss_G = alambda*loss_adv + alpha*loss_G_fake + beta*loss_perturb
            loss_G.backward()
            self.optimizer_G.step()
        # gia tri sai lech voi anh origin
        self.writer.add_scalar('iter/train/loss_D_real', loss_D_real.data, global_step=self.iteration)
        # gia tri sai lech voi anh fake
        self.writer.add_scalar('iter/train/loss_D_fake', loss_D_fake.data, global_step=self.iteration)
        # gia tri sai lech voi anh fake
        self.writer.add_scalar('iter/train/loss_G_fake', loss_G_fake.data, global_step=self.iteration)
        # noise gauss
        self.writer.add_scalar('iter/train/loss_perturb', loss_perturb.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_adv', loss_adv.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/loss_G', loss_G.data, global_step=self.iteration)
        self.writer.add_scalar('iter/train/fake_acc', fake_accuracy.data, global_step=self.iteration)
        self.iteration += 1
        
        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), loss_G.item(), fake_accuracy

    def train(self, train_dataloader, epochs):
        if self.iteration is None:
            self.iteration = (self.start_epoch-1)*len(train_dataloader)+1
       
        print("Starting training")
        for epoch in range(self.start_epoch, epochs+1):
            with open("./models/fool_rate_log.txt", 'a', encoding='utf-8') as file:
                file.write(str(epoch) + '\n')
            print("Start epoch num ", epoch)
            if epoch == self.epoch_of_change:
                self._use_attacker = False                        
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_G_sum = 0
            fake_acc_sum = 0
            
            for i, data in tqdm(enumerate(train_dataloader, start=0)):
                gc.collect()
                images,labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)  
         
                for j in range(0, self.batsize):
                    label_index = self.model(images[j], one_img=True)
                    if  label_index == None:
                        label_index = 80
                    labels[j] = label_index              
                # print('label', labels)
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_G_batch, fake_acc_batch = \
                      self.train_batch(images, labels)
                 
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_G_sum += loss_G_batch
                fake_acc_sum += fake_acc_batch 
                     
                if i % 50 == 0:
                    perturbation = self.netG(images)
                    self.writer.add_images('train/adversarial_perturbation', perturbation, global_step=epoch*len(train_dataloader)+i)
                    self.writer.add_images('train/adversarial_images', images+perturbation, global_step=epoch*len(train_dataloader)+i)
                    self.writer.add_images('train/adversarial_images_cl', torch.clamp(images+perturbation, self.box_min, self.box_max), global_step=epoch*len(train_dataloader)+i)
            
            # print statistics
            num_batch = len(train_dataloader)
            self.writer.add_scalar('epoch/train/loss_D', loss_D_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_G_fake', loss_G_fake_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_perturb', loss_perturb_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_adv', loss_adv_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/loss_G', loss_G_sum/num_batch, global_step=epoch)
            self.writer.add_scalar('epoch/train/fake_acc', fake_acc_sum/num_batch, global_step=epoch)
            
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, loss_G: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch, loss_G_sum/num_batch))

            # save generator
            if epoch%1==0:
                netG_file_name = self.netG_file_name + '.'+str(epoch)
                torch.save(self.netG.state_dict(), netG_file_name)
                netD_file_name = self.netDisc_file_name  + '.'+ str(epoch) 
                torch.save(self.netDisc.state_dict(), netD_file_name)
                optG_file_name = self.optG_file_name  + '.' + str(epoch) 
                torch.save(self.optimizer_G.state_dict(), optG_file_name)
                optD_file_name = self.optD_file_name  + '.' + str(epoch) 
                torch.save(self.optimizer_D.state_dict(), optD_file_name)
            
        #save final model
        torch.save(self.netG.state_dict(), self.netG_file_name )
        torch.save(self.netDisc.state_dict(), self.netDisc_file_name)
        torch.save(self.optimizer_G.state_dict(), self.optG_file_name)
        torch.save(self.optimizer_D.state_dict(), self.optD_file_name)