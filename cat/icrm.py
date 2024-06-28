import torch
import random
import torchvision
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from detectron2.structures.boxes import pairwise_ioa, pairwise_iou
import os
from detectron2.layers import cat
import detectron2.utils.comm as comm
import numpy as np



class ICRm:
    def __init__(self,num_classes = 8, max_save = 50, dir = "", blocked_classes = [],mix_ratio=0.5, cfg = None):
        self.num_classes = num_classes
        self.max_save = max_save
        self.b = torch.distributions.beta.Beta(0.5,0.5)
        self.class_info = torch.zeros(self.num_classes,self.num_classes)
        self.target_class_info = torch.zeros(self.num_classes,self.num_classes)
        self.alpha = 0.99
        self.target_flag = False
        self.blocked_classes = blocked_classes
        self.limit_mix = cfg.LIMIT_MIX
        self.keep_aspect_ratio = cfg.KEEP_ASPECT
        self.mix_ratio = mix_ratio
        self.cfg = cfg
        if cfg.MIX_RANDOM_CLASSES:
            self.mix_flag=False
        else:
            self.mix_flag=True

        self.dir = dir

        if 'last_checkpoint' in os.listdir(dir):
            with open(f'{dir}/last_checkpoint') as f:
                model_iter = f.read()
                iter = int(model_iter.split('_')[-1].split('.')[0])
                try:
                    self.iter = iter+1
                    self.class_info = torch.load(f'{self.dir}/prob_iter{iter}_rank{comm.get_local_rank()}.pt')
                    self.class_info = torch.load(f'{self.dir}/prob_target_iter{iter}_rank{comm.get_local_rank()}.pt')
                except:
                    try:
                        self.iter = iter
                        self.class_info = torch.load(f'{self.dir}/prob_iter{iter}_rank{comm.get_local_rank()}.pt')
                        self.class_info = torch.load(f'{self.dir}/prob_target_iter{iter}_rank{comm.get_local_rank()}.pt')
                    except:
                        pass

        

        # Bank is a list of lists [ []*number of classes ]. 
        # Each crop is a tuple (cropped image, bounding box tensor where corner is 0, ground truth class)
        self.bank = [[] for i in range(self.num_classes)]
        self.target_bank = [[] for i in range(self.num_classes)]

    def save_crops(self,label_data):
        for data in label_data:
            inst = data['instances']
            boxes = inst.gt_boxes.tensor.type(torch.long)
            gt_classes = inst.gt_classes
            
            image = data['image']
            #unique_classes = gt_classes.unique()
            #for u_cls in unique_classes:
            #    mask = gt_classes==u_cls
            #    box = boxes[mask][0]
            #    gt_cls = gt_classes[mask][0]
            #    cropped_image = image[:,box[1]:box[3],box[0]:box[2]]
            #    box[[0,2]]-=box[0]
            #    box[[1,3]]-=box[1]
            #    self.bank[gt_cls].append((cropped_image,box,gt_cls))

            for box, gt_cls in zip(boxes,gt_classes):
                cropped_image = image[:,box[1]:box[3],box[0]:box[2]]
                box[[0,2]]-=box[0]
                box[[1,3]]-=box[1]
                self.bank[gt_cls].append((cropped_image,box,gt_cls))

        for i in range(self.num_classes):
            if len(self.bank[i]) > self.max_save:
                self.bank[i] = self.bank[i][-self.max_save:]

    def save_crops_target(self,label_data):
        for data in label_data:
            inst = data['instances']
            boxes = inst.gt_boxes.tensor.type(torch.long)
            gt_classes = inst.gt_classes
            full_scores = inst.full_scores
            
            image = data['image']
            #unique_classes = gt_classes.unique()
            #for u_cls in unique_classes:
            #    mask = gt_classes==u_cls
            #    box = boxes[mask][0]
            #    gt_cls = gt_classes[mask][0]
            #    cropped_image = image[:,box[1]:box[3],box[0]:box[2]]
            #    box[[0,2]]-=box[0]
            #    box[[1,3]]-=box[1]
            #    self.bank[gt_cls].append((cropped_image,box,gt_cls))

            for box, gt_cls,full_score in zip(boxes,gt_classes,full_scores):
                if sum(full_score)<1:
                    continue
                cropped_image = image[:,box[1]:box[3],box[0]:box[2]]
                box[[0,2]]-=box[0]
                box[[1,3]]-=box[1]
                self.target_bank[gt_cls].append((cropped_image,box,gt_cls))

        for i in range(self.num_classes):
            if len(self.target_bank[i]) > self.max_save:
                self.target_bank[i] = self.target_bank[i][-self.max_save:]

    def mix_crop_new(self,label_data,target=False):

        # Determine which class info to use
        if target:
            if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP+1000:
                rat = (self.iter-self.cfg.SEMISUPNET.BURN_UP_STEP)/1000
                c_info = (1-rat) * self.class_info + rat * self.target_class_info
            else:
                c_info = self.target_class_info
        else:
            c_info = self.class_info
        # Mean of probability of a class being correctly predicted
        
        class_mean = np.mean([c_info[i,i] for i in range(len(c_info))])

        for data in label_data:

            #skip half of target samples
            #if (target and self.mix_ratio) or torch.rand(1)>0.5:
            #    flag_noaug = True
            #else:
            #    flag_noaug = False

            if torch.rand(1)<self.mix_ratio:
                flag_noaug = True
            else:
                flag_noaug = False

            # Extract gt_classes and bboxes for instances in the image
            inst = data['instances']
            gt_classes = inst.gt_classes
            boxes = inst.gt_boxes.tensor

            #Check if it is target or source
            if inst.has('full_scores'):
                full_scores = inst.full_scores
                box_loss_val = 0
            else:
                full_scores = None
                box_loss_val = 1

            oht = []
            box_loss = []
            new_boxes = []
            new_gt_classes = []
            all_full_scores = []
            image = data['image']
            minority_base=False

            for i, (box, gt_cls) in enumerate(zip(boxes,gt_classes)):
                
                # Add the original data into the new data
                new_boxes.append(box)
                new_gt_classes.append(gt_cls)
                box_loss.append(box_loss_val)
                if target:
                    if sum(full_scores[i])<1:
                        oht.append(full_scores[i])
                    else:
                        oht.append(torch.nn.functional.one_hot(gt_cls,self.num_classes+1)*1.0)
                    all_full_scores.append(full_scores[i])
                else:
                    oht.append(torch.nn.functional.one_hot(gt_cls,self.num_classes+1)*1.0)

                if flag_noaug:
                    continue

                # Determine if it is a majority or minority class based on the mean probability. 
                if c_info[gt_cls,gt_cls] > class_mean:
                    #Get the values from the class-relation table. Class itself should be remove. 
                    # Values represent the probality of a class bring predicted as the current GT class
                    values = c_info[:,gt_cls].clone()
                    values[gt_cls]=0
                else:
                    # Values represent the probality of the current GT class being represented a other classes
                    minority_base = True
                    values = c_info[gt_cls].clone()

                # We dont want to augment minority target images
                if minority_base and target:
                    continue

                # Don't allow certain classes to be copies onto image
                for blk_cls in self.blocked_classes:
                    values[blk_cls]=0

                # if values are too low we skip. Is this needed?
                if sum(values)<0.1:
                    continue


                # Use weighted sampling to get the mix class
                if self.mix_flag:
                    mix_cls = torch.multinomial(values, 1, replacement=True).to(gt_cls.device)
                else:
                    mix_cls = torch.randint(len(values),(1,1))[0].to(gt_cls.device)

                # Skip if the ground truth class is equal to the mix class.
                # We keep the GT class to have a chance to skip and not augment all the minorty data.
                if gt_cls == mix_cls:
                    continue
                
                use_target=False
                if self.target_flag and target:
                    if len(self.target_bank[mix_cls]) != 0:
                        use_target = True
                if minority_base or random.random()<0.2:
                    use_target = True

                # Use target image bank if it exists and using the minority base or a random chance
                if self.target_flag and use_target:
                    current_bank = self.target_bank
                else:
                    current_bank = self.bank

                # 50% change to skip current or if there is no samples
                if random.random()<0.5 or len(current_bank[mix_cls]) == 0:
                    continue
                
                # Randomly select a sample form the current image bank
                if len(current_bank[mix_cls])-1 != 0:
                    rand_idx = random.randint(0,len(current_bank[mix_cls])-1)
                else:
                    rand_idx = 0
                crop_img, crop_box, crop_cls = current_bank[mix_cls][rand_idx]

                # Get box values from the base instance
                boxl = box.type(torch.long)
                h = boxl[3]-boxl[1]
                w = boxl[2] - boxl[0]
                crop_box = crop_box.to(h.device)

                
                # Get mix-up values. We ensure mix_b is alows more than 5
                mix_b = self.b.sample()
                if mix_b>0.5 and self.limit_mix:
                    mix_b = 1 - mix_b

                if minority_base and mix_b<0.5 and self.limit_mix:
                    mix_b = 1 - mix_b

                # If mix box is too small we skip
                if crop_box[3]/h <0.4 or crop_box[2]/w <0.4:
                    continue

                if self.keep_aspect_ratio:
                    # Resize the crop_box based on the longest value
                    if crop_box[2]/w >= crop_box[3]/h:
                        crop_box_new = (crop_box*(1/(crop_box[2]/w))).type(torch.long)
                    else:
                        crop_box_new = (crop_box*(1/(crop_box[3]/h))).type(torch.long)

                    # Sometimes resizing fails?
                    try:
                        new_crop_img = torchvision.transforms.Resize((int(crop_box_new[3]),int(crop_box_new[2])))(crop_img)
                    except:
                        continue

                    # Calculate the height and width offsets
                    h_diff = torch.abs(h - new_crop_img.shape[1])
                    h_btm = torch.div(h_diff, 2, rounding_mode='floor')
                    h_top = h_diff - h_btm
    
                    w_diff = torch.abs(w - new_crop_img.shape[2])
                    w_left = torch.div(w_diff, 2, rounding_mode='floor')
                    w_right = w_diff - w_left
    
                        # Get the new box values
                    boxl = boxl + torch.tensor([w_left,h_top,-w_right,-h_btm]).to(h_btm.device)
    
                        # Mix-up images
                    image[:,boxl[1]:boxl[3],boxl[0]:boxl[2]] = (mix_b*image[:,boxl[1]:boxl[3],boxl[0]:boxl[2]] + (1-mix_b)*new_crop_img).type(torch.uint8)
    
                        # Add new mixed instance
                    new_boxes.append(boxl.type(torch.float))
                    if minority_base:
                        new_gt_classes.append(gt_cls)
                    else:
                        new_gt_classes.append(crop_cls)
                    oht.append(mix_b*torch.nn.functional.one_hot(gt_cls,self.num_classes+1)+(1-mix_b)*torch.nn.functional.one_hot(crop_cls.to(gt_cls.device),self.num_classes+1))
                    box_loss.append(1)
                    if target:
                        all_full_scores.append(torch.zeros_like(full_scores[i]))
                else:
                    # Sometimes resizing fails?
                    try:
                        new_crop_img = torchvision.transforms.Resize((int(h),int(w)))(crop_img)
                    except:
                        print('Resize failed....')
                        continue
                    # Mix-up images
                    image[:,boxl[1]:boxl[3],boxl[0]:boxl[2]] = (mix_b*image[:,boxl[1]:boxl[3],boxl[0]:boxl[2]] + (1-mix_b)*new_crop_img).type(torch.uint8)
                    oht[-1]=mix_b*torch.nn.functional.one_hot(gt_cls,self.num_classes+1)+(1-mix_b)*torch.nn.functional.one_hot(crop_cls.to(gt_cls.device),self.num_classes+1)
                    if minority_base:
                        new_gt_classes[-1] = gt_cls
                    else:
                        new_gt_classes[-1] = crop_cls

            # Create new instance object and add items
            new_instance = Instances(inst.image_size)
            new_instance.gt_classes = torch.tensor(new_gt_classes)
            try:
                new_instance.gt_boxes = Boxes(torch.vstack(new_boxes))
                new_instance.gt_oht_classes = torch.vstack(oht)
                new_instance.box_loss = torch.tensor(box_loss).to(gt_classes.device)
                if target:
                    new_instance.full_scores = torch.vstack(all_full_scores)
            except:
                new_instance.gt_boxes = inst.gt_boxes
                new_instance.gt_oht_classes = torch.tensor([]).to(gt_classes.device)
                new_instance.box_loss = torch.tensor([]).to(gt_classes.device)
                if target:
                    new_instance.full_scores = torch.tensor([]).to(gt_classes.device)

            data['instances'] = new_instance
            image = data['image']

        return label_data

    def add_labels(self,label_data):
        for data in label_data:
            inst = data['instances']
            gt_classes = inst.gt_classes
            boxes = inst.gt_boxes.tensor
            oht = []
            box_loss = []
            image = data['image']

            for box, gt_cls in zip(boxes,gt_classes):
                oht.append(torch.nn.functional.one_hot(gt_cls,self.num_classes+1)*1.0)
                box_loss.append(1)

            new_instance = Instances(inst.image_size)
            new_instance.gt_classes = gt_classes
            new_instance.gt_boxes = Boxes(boxes)
            try:
                new_instance.gt_oht_classes = torch.vstack(oht)
                new_instance.box_loss = torch.tensor(box_loss).to(gt_classes.device)
            except:
                new_instance.gt_oht_classes = torch.tensor([]).to(gt_classes.device)
                new_instance.box_loss = torch.tensor([]).to(gt_classes.device)

            data['instances'] = new_instance
            image = data['image']

        return label_data
    
    def match_boxes(self, gt_boxes, pred_boxes):
        iou = pairwise_iou(gt_boxes.to(pred_boxes.device),pred_boxes)
        try:
            max_iou = torch.max(iou,dim=0)
            p_idx = torch.where(max_iou[0]>0.5)[0]
            g_idx = max_iou[1][p_idx]
            pair = torch.vstack((g_idx,p_idx)).T
            return np.array(pair.cpu())
        except:
            return []
        
    def get_matches(self, proposals_predictions,iter, target=False):
        self.iter = iter
        proposals, predictions = proposals_predictions
        #proposals = proposals[:len(predictions)]
        pred = predictions[0].clone().cpu()
        gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)).cpu()
        del proposals
        del predictions

        pred_classes = torch.max(pred,dim=1)[1]

        # [x,0] is gt, [x,1] is predicted
        pairs = torch.vstack((gt_classes,pred_classes)).T

        #remove background 
        pairs = pairs[torch.all(pairs!=self.num_classes,dim=1)]

        # Rows represents GT, Columns represents predictions. GT=1, Pred=4 > [1,4]
        class_tensor = torch.zeros(self.num_classes,self.num_classes)
        class_tensor.index_put_(list(pairs.T),torch.tensor(1.0), accumulate=True)

        sum_values = torch.sum(class_tensor,dim=1)
        mask = sum_values!=0
        class_tensor[mask] = class_tensor[mask]/sum_values[mask].reshape(-1,1)

        if iter >= self.cfg.SEMISUPNET.BURN_UP_STEP:
            self.target_flag = True
            
        if iter < self.cfg.SEMISUPNET.BURN_UP_STEP/2:
            alpha = 0.5 + (self.alpha-0.5)*(iter/(self.cfg.SEMISUPNET.BURN_UP_STEP/2))**3
        else:
            alpha = self.alpha
        if target and iter < self.cfg.SEMISUPNET.BURN_UP_STEP*1.5:
            alpha = 0.5 + (self.alpha-0.5)*((iter-(self.cfg.SEMISUPNET.BURN_UP_STEP-1))/(self.cfg.SEMISUPNET.BURN_UP_STEP/2))**3

        if target:
            self.target_class_info[mask] = self.target_class_info[mask]*alpha + class_tensor[mask]*(1-alpha)
        else:
            self.class_info[mask] = self.class_info[mask]*alpha + class_tensor[mask]*(1-alpha)
        #self.class_bias = self.class_bias*alpha + bias_values*(1-alpha)

        if not (iter+1)%self.cfg.SOLVER.CHECKPOINT_PERIOD and iter:
            torch.save(self.class_info,f'{self.dir}/prob_iter{iter}_rank{comm.get_local_rank()}.pt')
            torch.save(self.target_class_info,f'{self.dir}/prob_target_iter{iter}_rank{comm.get_local_rank()}.pt')
        

