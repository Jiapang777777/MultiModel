# cross-attention

import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel4(nn.Module):
    def __init__(self, args):
        super(MultiModel4, self).__init__()
        self.TextModel = TextModel(args)
        self.ImageModel = ImageModel(args)

        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=2)

        # 多模态分类器_三层
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.hidden_size,args.middle_hidden_size),
            nn.ReLU(),
            nn.Linear(args.middle_hidden_size,3),
        )

    def forward(self, texts=None, imgs=None):
        # 多模态数据
        text_out = self.TextModel(texts)  # N, E
        img_out = self.ImageModel(imgs)  # N, E
       
        
        # 多头自注意力
        # 优化？要不要先加linear
        # Img --> Text
        x_i2t,_ = self.MultiheadAttention(text_out,img_out,img_out)
        # x_i2t2 = torch.mean(x_i2t,dim=0,keepdim=True).repeat(x_i2t.shape[0],1)
        # print(x_i2t2.shape)
        # print(x_i2t2)
        # Text --> Img
        x_t2i,_ = self.MultiheadAttention(img_out,text_out,text_out)
        # x_t2i2 = torch.mean(x_t2i, dim=0,keepdim=True).repeat(x_i2t.shape[0],1)
        # 组合
        # multi_out = torch.stack((x_i2t2, x_t2i2), 1)
        multi_out = torch.stack((x_i2t, x_t2i), 1)


        # 分类器
        multi_out = self.classifier_multi(multi_out)
        return multi_out
