import json
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from config import arguments

# 将所有数据存储为json格式
def transform(values):
    path_data = './datasets/data/'
    dataset = []
    for i in range(len(values)):
        guid = str(int(values[i][0]))
        tag = values[i][1]
        if type(tag) != str and math.isnan(tag):
            tag = None
        # print(tag)
        file = path_data + guid + '.txt'
        text = ''
        with open(file,'r',encoding='gb18030') as fp:
            for line in fp.readlines():
                line = line.strip('\n') #注意这里是把所有test合在了一起
                text += line
        dataset.append({
            'guid': guid,
            'tag': tag,
            'text': text,
            'img': path_data + guid + '.jpg',
        })
    return dataset


if __name__ == '__main__':
    dataset = pd.read_csv('./datasets/train.txt')
    # 分隔训练集、验证集
    train_data, dev_data = train_test_split(dataset, test_size=0.2)
    test_data = pd.read_csv('./datasets/test_without_label.txt')

    with open(arguments.train_file, 'w', encoding="utf-8") as f:
        train_set = transform(train_data.values)
        json.dump(train_set, f,indent=2) #注意加上indent=2，这样可以格式化

    with open(arguments.dev_file, 'w', encoding="utf-8") as f:
        dev_set = transform(dev_data.values)
        json.dump(dev_set, f,indent=2)

    with open(arguments.test_file, "w", encoding="utf-8") as f:
        test_set = transform(test_data.values)
        json.dump(test_set, f,indent=2)
