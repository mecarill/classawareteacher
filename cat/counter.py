from detectron2.structures.boxes import pairwise_iou
import torch
import numpy as np
import json
import os

class PredictionCounter:
    def __init__(self,dir):
        self.save_dir = dir+"/save_info/"
        os.makedirs(self.save_dir,exist_ok=True)

        self.reset_count = 0
        self.reset_iter = 10000
        self.total_count = {}
        self.pred_second_highest = {}
        self.pred_second_highest_below = {}
        self.gt_second_highest = {}
        self.gt_second_highest_below = {}
        for i in range(8):
            self.total_count[i]=[0,0,0,0,0,0,0,0]
            self.pred_second_highest[i]=[0,0,0,0,0,0,0,0]
            self.pred_second_highest_below[i]=[0,0,0,0,0,0,0,0]
            self.gt_second_highest[i]=[0,0,0,0,0,0,0,0]
            self.gt_second_highest_below[i]=[0,0,0,0,0,0,0,0]

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
    
    def check_reset(self,iter):
        if not iter % self.reset_iter:
            # SAVE
            files = {"total":self.total_count, "pred_second_highest":self.pred_second_highest, "pred_second_highest_below":self.pred_second_highest_below,"gt_second_highest":self.gt_second_highest,"gt_second_highest_below":self.gt_second_highest_below}
            with open(f"{self.save_dir}end_iter{iter}.json", "w") as outfile:
                json.dump(files, outfile)
            self.reset_count +=1
            for i in range(8):
                self.total_count[i]=[0,0,0,0,0,0,0,0]
                self.pred_second_highest[i]=[0,0,0,0,0,0,0,0]
                self.pred_second_highest_below[i]=[0,0,0,0,0,0,0,0]
                self.gt_second_highest[i]=[0,0,0,0,0,0,0,0]
                self.gt_second_highest_below[i]=[0,0,0,0,0,0,0,0]

    def get_matches(self, gt_instances, pred_instances):
        for gt, pred in zip(gt_instances, pred_instances):
            pairs = self.match_boxes(gt.gt_boxes, pred.pred_boxes)
            for pair in pairs:
                g_idx, p_idx = pair
                p_inst = pred[[p_idx]]
                g_inst = gt[[g_idx]]

                p_cls = np.array(p_inst.pred_classes.cpu())[0]
                g_cls = np.array(g_inst.gt_classes.cpu())[0]
                self.total_count[p_cls][g_cls] +=1

                if p_cls == g_cls:
                    continue
                p_second = torch.sort(p_inst.full_scores[0,:-1])[1][-2]
                p_second = np.array([p_second.cpu()])[0]
                self.pred_second_highest[p_cls][p_second] +=1
                self.gt_second_highest[g_cls][p_second] +=1

                if p_inst.scores[0]<0.8:
                    self.pred_second_highest_below[p_cls][p_second] +=1
                    self.gt_second_highest_below[g_cls][p_second] +=1





