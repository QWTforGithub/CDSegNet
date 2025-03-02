"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.utils.registry import Registry

LOSSES = Registry("losses")

class Criteria(object):
    def __init__(self, cfg=None, loss_type="EW", task_num=2):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))
        self.loss_type = loss_type
        self.task_num = task_num

    def __call__(self, points):

        if len(self.criteria) == 0:
            # loss computation occur in model
            return points

        loss = 0.0
        loss_mode = points["loss_mode"]

        if(loss_mode == "eval" or self.loss_type == "EW"):
            for c in self.criteria:
                l = c(points)
                loss += l

        elif(loss_mode == "train" and self.loss_type == "GLS"):
            loss = []
            for c in self.criteria:
                l = c(points)
                loss.append(l)

            if(self.task_num == 1):
                loss = loss[0] + loss[1]
            elif (self.task_num == 2 and self.task_num != len(loss)):
                loss = [loss[0], loss[1] + loss[2]] # MSE, Cross Entropy + Lovaz
                loss = loss[0] * loss[1]

            loss = torch.pow(loss, 1. / self.task_num)

        return loss

def build_criteria(cfg,loss_type="EW", task_num=2):
    return Criteria(cfg,loss_type=loss_type, task_num=task_num)
