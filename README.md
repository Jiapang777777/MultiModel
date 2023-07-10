# 实验五——多模态情感分类

该仓库储存了做此次实验所有相关的代码和数据

## 环境配置

环境依赖已经列在 requirements.txt 中了，使用

```py
pip install -r requirements.txt
```



## 文件内容介绍

```py
|-- 1.py         # 处理压缩包
|-- data.zip     # 本次实验原始数据的压缩包（因为超过了限制就没传上去）
|-- deal_data.py # 处理解压后的原始数据
|-- checkpoints  #文件夹，用来保存在运行过程中的模型
|-- datasets     #文件夹，用来放置deal_data处理后的文件
    |-- data #原始数据
    |-- dev.json  #处理后的验证集
    |-- test_without_label.txt #需要预测的标签文件
    |-- test.json # 处理后的测试集
    |-- train.json #处理后的训练集
|-- Model # 所有模型
    |-- ImageModel.py  #图像模型
    |-- MultiModel1.py #方法一的多模态模型
    |-- MultiModel2.py #方法二的多模态模型
    |-- MultiModel3.py #方法三的多模态模型
    |-- MultiModel4.py #方法四的多模态模型
    |-- MultiModel5.py #方法五的多模态模型
    |-- TextModel.py   #文本模型
|-- pre_model # 预训练模型Roberta
    |-- config.json
    |-- merges.txt
    |-- pytorch_model.bin #（因为超过了限制就没传上去）
    |-- tockenizer.json
    |-- vocab.json
|-- config.py # 实验中所需要的参数
|-- data_utils.py # 处理数据为符合模型的形式
|-- function.py # 训练、验证、预测、画图等辅助函数
|-- main.py # 主函数（训练、预测、画图一条龙）
|-- requirements.txt # 环境依赖
|-- test_with_label.txt # 预训练模型Roberta
|-- README.md # 本文件
```



## 运行方法

（因为文件受大小限制，需手动在huggingface上下载Roberta的文件 [pytorch_model.bin](pre_model/pytorch_model.bin) 到pre_model文件夹）

先配置环境依赖：

```
pip install -r requirements.txt
```

- 如果datasets文件夹为空，在根目录下只拥有data.zip压缩包，那么先依次运行

```py
python 1.py
python deal_data.py
```

- 如果datasets文件夹有以上文件，则可以开始直接运行模型

  消融实验（仅文本0.7137）

  ```py
  python main.py --model_name MultiModel1 --choose_class text --lr 2e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  ```

  消融实验（仅图像）

  ```
  python main.py --model_name MultiModel1 --choose_class img --lr 2e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  ```

  多模态试验（有五个方法）

  ```py
  # 模型一——0.7325
  python main.py --model_name MultiModel1  --lr 1e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  # 模型二——0.6019
  python main.py --model_name MultiModel2 --lr 1e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  # 模型三——0.7475
  python main.py --model_name MultiModel3 --lr 9e-6 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  # 模型四——0.7037
  python main.py --model_name MultiModel4 --lr 1e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 5
  # 模型五——0.7025
  python main.py --model_name MultiModel5 --lr 1e-5 --hidden_size 1000 --middle_hidden_size 500 --epoch 3
  ```



## 参数选择

```py
--model_name （MultiModel1、MultiModel2、MultiModel3、MultiModel4、MultiModel5）
--choose_class （text、img、multi）
--epoch
--batch_size
--dropout
--hidden_size
--middle_hidden_size
...
```



## 实验结果汇总

##### 多模态

| Model                                                    | ACC        |
| :------------------------------------------------------- | :--------- |
| **方法一**：简单特征拼接                                 | 0.7325     |
| 方法二：加上自注意力层                                   | 0.6019     |
| **方法三**：在方法二基础上对特征用transformerencoder封装 | **0.7475** |
| 方法四：对特征使用交叉注意力机制                         | 0.7037     |
| 方法五：交叉注意力机制+文本特征拼接                      | 0.7025     |

##### 消融实验

|  仅Text   | 0.7173 |
| :-------: | :----: |
| **仅Img** | 0.6388 |



## 实验参考

多模态情感分析vistanet实践：https://zhuanlan.zhihu.com/p/345713441

多模态条件机制：https://mp.weixin.qq.com/s?__biz=Mzk0MzIzODM5MA==&mid=2247486441&idx=1&sn=06df067828b19ef9aeef99f455f897e9&chksm=c337b670f4403f663f7b98a2aa75cb5062bf5a6222c81ce8f181d79d367971a4587b62da84a1#rd

self-cross-attention：https://github.com/smartcameras/SelfCrossAttn

图文匹配：https://zhuanlan.zhihu.com/p/565986474

vistanet代码实现：https://github.com/PreferredAI/vista-net