from torch import nn
from transformers import RobertaModel

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.bert = RobertaModel.from_pretrained('pre_model/')
        
        self.trans = nn.Sequential(
            # 注意bert的输出就是768
            # nn.Dropout(p=args.dropout),
            nn.Linear(768, args.hidden_size),
            nn.ReLU(),
        )
        # 此处用于调节是否将BERT纳入微调训练fine-tune
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, text_input):
        # print(text_input) 由 'input_dis' 'attention_mask' ''
        # **text_input 就是 input_ids
        # 在这里补充说明以下text_input是一个字典，保存了两个向量。
        bert_out = self.bert(**text_input)
        # pooler_output是序列的最后一层的隐藏状态的第一个token
        pooler_output = bert_out['pooler_output']
        # N:bach大小, L:序列长度, E:特征维度
        return self.trans(pooler_output)

