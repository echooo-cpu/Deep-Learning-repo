import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import yaml
import logging
import argparse


def load_dataset():


    diabets = load_diabetes()
    X , y = diabets.data, diabets.target

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size = 1/6, random_seed = 42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size = 1/5, random_seed = 42)

    print(f"train dataset size: {x_train.shape} valid dataset size: {x_valid.shape} test dataset size: {x_test.shape}")
    return x_train, y_train, x_valid, y_valid, x_test, y_test

class fnn(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.fnn_class == config['model']['fnn_class']

        if config['model']['activation'] == 'relu':
            self.activation = nn.ReLU()
        elif config['model']['activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif config['model']['activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif config['model']['activation'] == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif config['model']['activation'] == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"unknown activation: {config['model']['activation']}")

        if self.fnn_class == 'small':
            self.net = nn.Sequential(
                nn.linear(10, config['model']['hidden_dim']),
                self.activation,
                nn.linear(config['model']['hidden_dim'], 1)
            )
        elif self.fnn_class == 'medium':
            self.net = nn.Sequential(
                nn.linear(10, config['model']['hidden_dim1']),
                self.activation,
                nn.linear(config['model']['hidden_dim1'], config['model']['hidden_dim2']),
                self.activation,
                nn.linear(config['model']['hidden_dim2'], 1)
            )
        elif self.fnn_class == 'large':
            self.net = nn.Sequential(
                nn.linear(10, config['model']['hidden_dim1']),
                self.activation,
                nn.linear(config['model']['hidden_dim1'], config['model']['hidden_dim2']),
                self.activation,
                nn.linear(config['model']['hidden_dim2'], config['model']['hidden_dim3']),
                self.activation,
                nn.linear(config['model']['hidden_dim3'], 1)
            )
        else:
            raise ValueError(f"unkown fnn_class: {self.fnn_class}")

    
    def forward(self,x):
        return self.net(x)
    
def main():

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="FNN 训练脚本")
    parser.add_argument('--config', type=str, default='config.yaml', help='YAML 配置文件的路径')
    args = parser.parse_args()


    with open(f'../cinfig/{args.config}', 'r') as f:
        config = yaml.safe_load(f)
    
    log_file_path = f"../result/fnn_{config['model']['fnn_class']}_{config['model']['activation']}_{config['learning_rate']}.log"

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        ]
    )
    

if __name__ == "__main__":

    main()
