# self-attention

import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel2(nn.Module):
    def __init__(self, args):
        super(MultiModel2, self).__init__()
        self.TextModel = TextModel(args)
        self.ImageModel = ImageModel(args)

        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=4)

        # 多模态分类器_三层
        self.classifier = nn.Sequential(
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
        # 文本
        # Q是Query，是输入的信息，即当前任务的目标，用于和key进行匹配；
        # K和V分别是Key和Value，一般是相同的数据，比如原始文本经过Embedding后的表征；
        x_txt,_ = self.MultiheadAttention(text_out,text_out,text_out) # 分别对应query、key、value
        # x_txt2 = torch.mean(x_txt,dim=0,keepdim=True).repeat(x_txt.shape[0],1)
        # print(x_txt.shape)
        # 图片
        x_img,_ = self.MultiheadAttention(img_out,img_out,img_out)
        # x_img2 = torch.mean(x_img,dim=0,keepdim=True).repeat(x_img.shape[0],1)
        # print(x_img2)
        # 组合
        multi_out = torch.stack((x_txt, x_img), dim=1)  

        # 分类器
        multi_out = self.classifier(multi_out)
        return multi_out
