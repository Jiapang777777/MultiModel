import torch
import torch.nn as nn
from torchvision.models import resnet152


class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.image_res = resnet152(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.image_res.children())[:-1]),
            nn.Flatten()
        )

        self.trans = nn.Sequential(
            # nn.Dropout(p=args.dropout),
            nn.Linear(self.image_res.fc.in_features, args.hidden_size),
            nn.ReLU(inplace=True)
        )
        
        for param in self.image_res.parameters():
            param.requires_grad = True

    def forward(self, imgs):
        feature = self.resnet(imgs)
        return self.trans(feature)

