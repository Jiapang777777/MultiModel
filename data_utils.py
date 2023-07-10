import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms #用于常见的图形变换
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from config import arguments


class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = RobertaTokenizer.from_pretrained('pre_model/',use_fast=True) #用提前下载好的
        # 对图像进行预处理
        # self.transform =  transforms.Compose(
        # [
        #  transforms.Resize((arguments.img_size, arguments.img_size)),
        #  transforms.ToTensor(),
        #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] #这里用的是pytorch上给的通用的统计值：
        # )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),#不转换为PIL会报错
            transforms.Resize((224,224)),  #缩放
            transforms.CenterCrop(224),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 转为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  #标准化

        self.map1 = {'positive':0,'negative':1,'neutral':2} #处理数据将tag转换为数字
        self.map2 = {0:'positive',1:'negative',2:'neutral'} #将预测得到的数字转换为tag

    def __len__(self): #必须重写——获取数据集的长度
        return len(self.data)

    def __getitem__(self, item): #获取数据集中的数据
        # 对每一条数据进行处理
        item = self.data[item]
        # 根据json格式来获取每一个特定的特征
        item_id = item['id']
        text = item['text']
        img = item['img'] # 这里相当于是图片的路径
        tag = item['tag']

        # text数据
        # truncation为True会把过长的输入切掉，从而保证所有的句子都是相同长度的，return_tensors=”pt”表示返回的是PyTorch的Tensor，如果使用TensorFlow则tf。
        input_text = self.tokenizer(text,max_length=64, padding='max_length', return_tensors="pt",truncation=True)
        input_text['attention_mask'] = input_text['attention_mask'].squeeze() # 真实的文本为1，其他为padding
        input_text['input_ids'] = input_text['input_ids'].squeeze() #input_text['input_ids'] 则是讲每一个text转换为对应的下标
        
        
        # img数据
        input_img = self.transform(img)
        
        # 处理训练集
        if tag in self.map1:
            input_tag = self.map1[tag] #这里就是把对应的情绪转换为标签
        # 处理测试集的为null
        else:
            input_tag = 3 #只用到了0，1，2

        return item_id, input_text, input_img, input_tag
        

# flag来决定是否shuffle 顺序是否打乱
def to_dataloager(file,flag):
    # 先转换成list
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for _, item in enumerate(items):
            data_item = {}
            data_item['id'] = item['guid']
            data_item['text'] = item['text']
            data_item['img'] = np.array(Image.open(item['img']))
            data_item['tag'] = item['tag']
            data_list.append(data_item)
    # 再转换成dataloader
    set = TensorDataset(data_list)
    # 训练需要打乱数据
    if flag == 'True':
        dataloader = DataLoader(set, shuffle=True, batch_size=arguments.batch_size)
        return dataloader
    # 测试则不需要打乱数据
    if flag == 'False':
        dataloader = DataLoader(set, shuffle=False, batch_size=arguments.batch_size)
        return dataloader
    

        

