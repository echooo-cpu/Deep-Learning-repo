import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import argparse
from tqdm import tqdm


# ...existing code...

def load_dataset():

    diabets = load_diabetes()
    X , y = diabets.data, diabets.target

    # 注意这里的参数名应为 random_state
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size = 1/6, random_state = 42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size = 1/5, random_state = 42)

    # 1. 初始化标准化器
    scaler = StandardScaler()
    
    # 2. 仅在训练集上 fit，并进行 transform
    x_train = scaler.fit_transform(x_train)
    
    # 3. 使用训练集得到的均值和方差来 transform 验证集和测试集
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # 4. 转换为 PyTorch 的张量格式 (顺便把 y_test 等转换为列向量以适配神经网络)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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
    parser.add_argument('batch_size', type=int, help='训练的批次大小')
    parser.add_argument('optimizer', type=str, help='优化器类型，例如 adam 或 sgd')
    args = parser.parse_args()

    max_epochs = 100


    with open(f'../config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)
    
    
    log_name = f"fnn_{config['model']['fnn_class']}_{config['model']['activation']} \
        _{config['learning_rate']}_bsz{args.batch_size}_opt{args.optimizer}"
    log_file_path = f"../result/{log_name}"
    os.makedirs(log_file_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_file_path}/{log_name}.log", mode='w', encoding='utf-8'),
        ]
    )

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset()

    model = fnn(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) if args.optimizer.lower() == 'adam' \
          else torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

    logging.info(f"config: {config}")

    valid_loss_history = []
    valid_loss_history.append(float('inf'))  # 初始化验证损失历史，初始值为无穷大

    for epoch in tqdm(range(max_epochs)):
        logging.info(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        pred = model(x_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_pred = model(x_valid)
        valid_loss = criterion(valid_pred, y_valid) 
        valid_loss_history.append(valid_loss.item())

        if valid_loss_history[-1] == min(valid_loss_history):
            torch.save(model.state_dict(), f"{log_file_path}/model_{log_name}.pth")
            break




    

if __name__ == "__main__":

    main()
