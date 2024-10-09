from __future__ import division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput

import scipy.sparse as sp
import numpy as np
import scipy.io as scio


class Multihead_Attention(nn.Module):    
    """
    multihead_attention
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 num_heads=1,                   
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)   
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_out = nn.Linear(num_heads, 1)

    def forward(self,
                Q, K):
        """
        :param Q: A 3d tensor with shape of [T_q, C_q]   
        :param K: A 3d tensor with shape of [T_k, C_k]   
        :param V: A 3d tensor with shape of [T_v, C_v]   
        :return:
        """
        num_heads = self.num_heads
        N = Q.shape[0]                                      
        # Q = Q.unsqueeze(dim = 0)        
        # K = K.unsqueeze(dim = 0)

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))                         
        K_l = nn.ReLU()(self.linear_K(K))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2) 
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)                   
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))    #(h*N, T_q(1), T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)   

        # Dropouts   
        outputs = self.dropout(outputs) 
        outputs = outputs.split(N, dim=0)
        outputs = torch.cat(outputs, dim=1)  #(bs, num_heads, num_point)
        outputs = outputs.transpose(1,2)     #(bs, num_point, num_heads)
        outputs = self.linear_out(outputs)   ##(bs, num_point, 1)
        outputs = nn.Softmax(dim=1)(outputs) 

        return outputs   


class ST_Pretrain(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(ST_Pretrain, self).__init__()

        self.image_size = 300
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(262, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        pointwise_in_channels = 64 + self.num_cate

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)


        self.dropout_rate = 0.35
        self.dropout = nn.Dropout(p=self.dropout_rate)

        
       
        self.target_object_attention = torch.nn.Parameter(torch.FloatTensor(self.num_cate, self.num_cate), requires_grad=True)
        self.target_object_attention.data.fill_(1/22)  

    
        self.scene_object_attention = torch.nn.Parameter(torch.FloatTensor(4, self.num_cate, self.num_cate), requires_grad=True)
        self.scene_object_attention.data.fill_(1/22) 
     
        self.attention_weight = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=True)
        self.attention_weight.data.fill_(1/2)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))    #(64,7,7) -> (64,1,1)
        # self.image_to_attent = nn.Linear(64,22)

      
        self.muti_head_attention = Multihead_Attention(hidden_dim = 512, C_q = resnet_embedding_sz + 64, C_k = 262, num_heads = 8, dropout_rate = 0.3)
        self.conf_threshod = 0.6
       
        self.num_cate_embed = nn.Sequential(
            nn.Linear(self.num_cate, 32), 
            nn.ReLU(),
            nn.Linear(32, 64),  
            nn.ReLU(),
        )

        self.pretrain_fc = nn.Linear(3136, 6)

    def one_hot(self, spa): 

        y = torch.arange(spa).unsqueeze(-1)   
        y_onehot = torch.FloatTensor(spa, spa) 

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)   

        return y_onehot   ## (22,22)


    def forward(self,global_feature: torch.Tensor, local_feature: dict):
        
        target_object = local_feature['indicator']                    # target object one-hot encode
        batch_size = global_feature.shape[0]        
        global_feature = global_feature.squeeze(dim=1)                # bs * 512 * 7 * 7 
        # print("global_feature shape:",  global_feature.shape)
        image_embedding = F.relu(self.conv1(global_feature))  

        x = self.dropout(image_embedding)

        ################################ compute object embedding #######################
        target_appear = local_feature['features']    # bs * 22 * dim
        target_conf = local_feature['scores'].unsqueeze(dim=2)  #bs * 22 * 1
        target_bbox = local_feature['bboxes'] / self.image_size
        # print('target_appear shape:', target_appear.shape)
        target = torch.cat((target_appear, target_bbox, target_conf, target_object), dim=2)  

        target_object_attention = F.softmax(self.target_object_attention, 0)        

        attention_weight = F.softmax(self.attention_weight, 0)    

        object_attention = target_object_attention * attention_weight[0]
      
        
        object_select = torch.sign(target_conf - 0.6)  
        object_select[object_select > 0] = 0                       
        object_select[object_select < 0] = - object_select[object_select < 0]   #(bs,22, 1)     
        object_select_appear = object_select.expand(batch_size, 22, 262).bool()          
        target_mutiHead = target.masked_fill(object_select_appear,0)            

        image_object_attention = self.avgpool(global_feature).squeeze(dim = 3).transpose(1,2)   #(bs, 1, 512)  
        spa = self.one_hot(self.num_cate).to(target.device).view(1, self.num_cate, self.num_cate).expand(batch_size, self.num_cate, self.num_cate)     
        num_cate_index = self.num_cate_embed(num_cate_index)   
        # num_cate_index = num_cate_index.view(1, 1, 64).repeat(batch_size, 1, 1)       # (bs, 1, 64)
        image_object_attention = torch.cat((image_object_attention, num_cate_index), dim = 2)  #(bs, 1, 512+64=576)
        image_object_attention = self.muti_head_attention(image_object_attention, target_mutiHead)  #(bs, num_point, 1)
        

        target_attention= torch.bmm(object_attention.view(1, self.num_cate, self.num_cate).expand(batch_size, self.num_cate, self.num_cate), target_object)  
        target_attention = target_attention + image_object_attention * attention_weight[1]  

        
        target = F.relu(self.graph_detection_feature(target))    #518-128维-49  N*49
        target = target * target_attention                       
        target_embedding = target.reshape(batch_size, self.num_cate, 7, 7)    # bs*N*7*7
        target_embedding = self.dropout(target_embedding)  
        ##############################################################################################################################


        x = torch.cat((x, target_embedding), dim=1)
        x = F.relu(self.pointwise(x))  #dim = 64
        x = self.dropout(x)
        out = x.view(x.size(0), -1)    #bs*64*7*7 to  bs*3136
        action = self.pretrain_fc(out)

        return {
            'action': action,
            'fc_weights': self.pretrain_fc.weight,
            'visual_reps': out.reshape(batch_size, 64, 49)
        }
