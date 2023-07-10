import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel3(nn.Module):
    def __init__(self, args):
        super(MultiModel3, self).__init__()
        self.TextModel = TextModel(args)
        self.ImageModel = ImageModel(args)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=2)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size*2, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # 多模态分类器_三层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            # nn.Linear(args.hidden_size,3)
            nn.Linear(args.hidden_size,args.middle_hidden_size ),
            nn.ReLU(),
            nn.Linear(args.middle_hidden_size,3)
        )

    def forward(self, texts=None, imgs=None):
        # 多模态数据
        text_out = self.TextModel(texts)  # [batch_size,1000]
        # print(text_out.shape)
        img_out = self.ImageModel(imgs)  # [batch_size,1000]
        # print(img_out.shape)
       
        # 使用封装的encoder
        # print(multi_middle.shape) 
        # multi_middle = torch.cat((text_out, img_out), dim=1)
        # 这里使用stack而不是cat，stack是同维度上叠加，注意可以尝试一下使用cat，然后self.transformer_encoder输入hidden要✖️2
        # multi_middle = torch.stack((text_out, img_out), dim=1)  # [batch_size,2,1000]
        # multi_out = self.transformer_encoder(multi_middle)

        multi_out1 = self.transformer_encoder(text_out)
        multi_out2 = self.transformer_encoder(img_out)
        multi_out = torch.cat((multi_out1,multi_out2), dim=1)

        # 分类器
        multi_out = self.classifier(multi_out)
        return multi_out



    
