import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel1(nn.Module):
    def __init__(self, args):
        super(MultiModel1, self).__init__()
        self.TextModel = TextModel(args)
        self.ImageModel = ImageModel(args)

        # 多模态分类器
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.hidden_size,args.middle_hidden_size ),
            nn.ReLU(),
            nn.Linear(args.middle_hidden_size,3),
        )

        # 单模态分类器
        self.classifier_single = nn.Sequential(
            nn.Linear(args.hidden_size, args.middle_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.middle_hidden_size, 3)
        )

    def forward(self, texts=None, imgs=None):
        # 仅文本
        if texts is None:
            img_out = self.ImageModel(imgs)
            img_out = self.classifier_single(img_out)
            return img_out

        # 仅图片
        if imgs is None:
            text_out = self.TextModel(texts)
            text_out = self.classifier_single(text_out)
            return text_out

        # 多模态数据
        text_out = self.TextModel(texts)  # N, E
        img_out = self.ImageModel(imgs)  # N, E
       
        # 直接将特征来进行拼接来进行分类
        multi_out = torch.cat((text_out, img_out), 1)

        # 分类器
        multi_out = self.classifier_multi(multi_out)
        return multi_out
