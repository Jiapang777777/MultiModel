import argparse
parser = argparse.ArgumentParser()

# 模型选择
parser.add_argument("-model_name", "--model_name",
                    type=str, default='MultiModel1', help='model_name') 
parser.add_argument("-choose_class", "--choose_class",
                    type=str, default='multi', help='choose_class')

# 数据集
parser.add_argument('-train_file', '--train_file',
                    type=str, default='./datasets/train.json', help='train_file')
parser.add_argument('-dev_file', '--dev_file',
                    type=str, default='./datasets/dev.json', help='dev_file')
parser.add_argument('-test_file', '--test_file',
                    type=str, default='./datasets/test.json', help='test_file')

# 模型、结果保存
parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
                    type=str, default='./checkpoints', help='output_dir')
parser.add_argument('-test_result', '--test_result',
                    type=str, default='./test_with_label.txt', help='test_result')

# 训练时用到的参数
parser.add_argument("-epoch", "--epoch",
                    type=int, default=5, help='epoch')
parser.add_argument("-batch_size", "--batch_size",
                    type=int, default=4, help='bach size')
parser.add_argument("-lr", "--lr",
                    type=float, default=1e-6, help='learning rate')
parser.add_argument("-dropout", "--dropout",
                    type=float, default=0.0, help='dropout')

# 模型内部用到的参数
parser.add_argument("-hidden_size", "--hidden_size",
                    type=int, default=800, help='hidden_size')
parser.add_argument("-middle_hidden_size", "--middle_hidden_size",
                    type=int, default=400, help='middle_hidden_size')

arguments = parser.parse_args()
