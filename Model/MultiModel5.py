# cross-attention
import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel5(nn.Module):
    def __init__(self, args):
        super(MultiModel5, self).__init__()
        self.TextModel = TextModel(args)
        self.ImageModel = ImageModel(args)

        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=2)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size*2, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

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
       
        # 图片自己的自注意力机制
        text_self_attention,_ = self.MultiheadAttention(text_out,text_out,text_out)
        # Text_self_attention + Img -- cross_attention
        text_cross_img,_ = self.MultiheadAttention(text_self_attention,img_out,img_out)
        multi_out = torch.stack((text_out, text_cross_img), 1)

        # 分类器
        multi_out = self.classifier_multi(multi_out)
        return multi_out
