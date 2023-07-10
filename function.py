import torch
from torch import nn
from config import arguments
import matplotlib.pyplot as plt

def picture(train_loss,train_acc,dev_acc):
    plt.subplot(1, 3, 1) #共两行三列，在第一个位置绘画
    plt.plot(train_loss)
    plt.title('Train Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label='val')
    plt.title('Train Accauray')
    plt.xlabel('Epoch')
    plt.ylabel('acc')

    plt.subplot(1, 3, 3) #共两行三列，在第一个位置绘画
    plt.plot(dev_acc)
    plt.title('Dev Acc')
    plt.xlabel('Epoch')
    plt.ylabel('acc')

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('1',dpi=72)
    print("ok")

def train(model,optimizer,loss_function,epochs,train_dataloader,dev_dataloader):
    train_loss,train_acc,dev_acc = [],[],[]
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_acc_total,train_loss_total = 0.0, 0.0
        total = 0
        for step, batch_datas in enumerate(train_dataloader):
            _, texts, imgs, label = batch_datas
            texts = texts.to(device=arguments.device)
            imgs = imgs.to(device=arguments.device)
            label = label.to(device=arguments.device)
            total += label.shape[0]
            
            # 训练模型
            if arguments.choose_class == 'multi':
                output = model(texts=texts, imgs=imgs)
            elif arguments.choose_class == 'text':
                output = model(texts=texts, imgs=None)
            elif arguments.choose_class == 'img':
                output = model(texts=None, imgs=imgs)

            # 计算损失
            loss = loss_function(output, label.long()).sum()

            # 计算train的loss和acc
            train_loss_total += loss.item()
            train_acc_total += (output.argmax(dim=1) == label).float().sum().item()
    
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch %d, loss %.4f, train acc %.4f' % (epoch, train_loss_total / total, train_acc_total / total))  
        # 计算验证集准确率
        acc = evaluate(model, dev_dataloader, epoch)
        # 为了画图传入数组
        train_loss.append(train_loss_total / total)
        train_acc.append(train_acc_total / total)
        dev_acc.append(acc)
        
        #保存模型
        if best_accuracy < acc:
            best_accuracy = acc
            torch.save(model.state_dict(), arguments.checkpoints_dir  + '/best_checkpoint.pth')

    return train_loss,train_acc,dev_acc
    
def evaluate(model, dev_dataloader, epoch=None):
    # 计算验证集准确率
    acc = 0.0
    total = 0
    for step, batch_datas in enumerate(dev_dataloader):
        ids, texts, imgs, y = batch_datas
        texts = texts.to(device=arguments.device)
        imgs = imgs.to(device=arguments.device)
        y = y.to(device=arguments.device)
        total += y.shape[0]
        # 得到结果
        if arguments.choose_class == 'multi':
            output = model(texts=texts, imgs=imgs)
        elif arguments.choose_class == 'text':
            output = model(texts=texts, imgs=None)
        elif arguments.choose_class == 'img':
            output = model(texts=None, imgs=imgs)
        # 算正确率
        acc += (output.argmax(dim=1) == y).float().sum().item()
        
    print('epoch %d, dev acc %.4f'% (epoch, acc / total))
    return acc / total


def predict(model,test_dataloader):
    # 先加载模型
    model.load_state_dict(torch.load(arguments.checkpoints_dir + '/best_checkpoint.pth', map_location=torch.device(device=arguments.device)))
    # 测试集预测
    print('predicting...')
    predict_list = []
    for step, batch_datas in enumerate(test_dataloader):
        ids, texts, imgs, y = batch_datas
        texts = texts.to(device=arguments.device)
        imgs = imgs.to(device=arguments.device)
        muli_label = model(texts=texts, imgs=imgs)
        predict_y = muli_label.argmax(dim=1)  # 使用主分类器
        for i in range(len(ids)):
            item_id = ids[i]
            # 这里是将数字转换成标签
            tag = test_dataloader.dataset.map2[int(predict_y[i])]
            predict_dict = {
                'guid': item_id,
                'tag': tag,
            }
            predict_list.append(predict_dict)
    with open(arguments.test_result, 'w', encoding='utf-8') as f:
        f.write('guid,tag' + '\n')
        for i in range(len(predict_list)):
            f.write(predict_list[i]['guid'])
            f.write(',')
            f.write(predict_list[i]['tag'])
            f.write('\n')

