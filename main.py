import torch
from torch import nn
from config import arguments
from data_utils import to_dataloager
from function import train,predict,picture
from Model.MultiModel1 import MultiModel1
from Model.MultiModel2 import MultiModel2
from Model.MultiModel3 import MultiModel3
from Model.MultiModel4 import MultiModel4
from Model.MultiModel5 import MultiModel5

Model_Set = {'MultiModel1': MultiModel1(arguments), 'MultiModel2': MultiModel2(arguments), 'MultiModel3': MultiModel3(arguments), 'MultiModel4': MultiModel4(arguments), 'MultiModel5': MultiModel5(arguments)}

if __name__ == '__main__':
    # cuda
    arguments.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('my device is:' + str(arguments.device))

    # 加载数据集
    train_dataloader = to_dataloager(arguments.train_file,'True')
    dev_dataloader = to_dataloager(arguments.dev_file,'True')
    test_dataloader = to_dataloager(arguments.test_file,'False')

    # 定义模型、分类器、损失函数、epoch
    print('training...')
    model = Model_Set[arguments.model_name].to(device=arguments.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)
    loss_function = nn.CrossEntropyLoss()
    epochs = arguments.epoch

    train_loss,train_acc,dev_acc = train(model,optimizer,loss_function,epochs,train_dataloader,dev_dataloader)
    predict(model,test_dataloader)
    
    #画图
    picture(train_loss,train_acc,dev_acc)
