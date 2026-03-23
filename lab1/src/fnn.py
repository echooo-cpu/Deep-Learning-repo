import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增：用于绘图

def load_dataset():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size=1/6, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=1/5, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    
    # 使用训练集得到的均值和方差来 transform 验证集和测试集
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # 转换为 PyTorch 的张量格式 (顺便把 y_test 等转换为列向量以适配神经网络)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f"Dataset sizes -> Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_valid, y_valid, x_test, y_test


class FNN(nn.Module):
    def __init__(self, fnn_class, activation_name, hidden_dims):
        super().__init__()
        self.fnn_class = fnn_class

        # 使用字典简化激活函数的选择
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'swish': nn.SiLU()
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        self.activation = activations[activation_name]

        # 确保传入的 hidden_dims 数量足够支撑对应的网络规模
        h_dims = hidden_dims + [10] * (3 - len(hidden_dims)) # 补齐防止索引越界

        if self.fnn_class == 'small':
            self.net = nn.Sequential(
                nn.Linear(10, h_dims[0]),
                self.activation,
                nn.Linear(h_dims[0], 1)
            )
        elif self.fnn_class == 'medium':
            self.net = nn.Sequential(
                nn.Linear(10, h_dims[0]),
                self.activation,
                nn.Linear(h_dims[0], h_dims[1]),
                self.activation,
                nn.Linear(h_dims[1], 1)
            )
        elif self.fnn_class == 'large':
            self.net = nn.Sequential(
                nn.Linear(10, h_dims[0]),
                self.activation,
                nn.Linear(h_dims[0], h_dims[1]),
                self.activation,
                nn.Linear(h_dims[1], h_dims[2]),
                self.activation,
                nn.Linear(h_dims[2], 1)
            )
        else:
            raise ValueError(f"Unknown fnn_class: {self.fnn_class}")

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="FNN 训练脚本")
    parser.add_argument('--fnn_class', type=str, default='small', choices=['small', 'medium', 'large'], help='网络规模')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid', 'leaky_relu', 'swish'], help='激活函数')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16], help='隐藏层维度列表，如: 64 32 16')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=5, help='早停容忍度')
    args = parser.parse_args()

    # 动态生成日志和保存路径
    log_name = f"fnn_{args.fnn_class}_{args.activation}_lr{args.lr}_bsz{args.batch_size}_opt{args.optimizer}"
    log_dir = os.path.join("..", "result", log_name)
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{log_name}.log"), mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Training Arguments: {vars(args)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    
    x_valid, y_valid = x_valid.to(device), y_valid.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = FNN(fnn_class=args.fnn_class, activation_name=args.activation, hidden_dims=args.hidden_dims).to(device)
    criterion = nn.MSELoss()
    
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)
        
        epoch_train_loss /= len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            valid_pred = model(x_valid)
            valid_loss = criterion(valid_pred, y_valid) 

        current_valid_loss = valid_loss.item()
        history['valid_loss'].append(current_valid_loss)
        
        logging.info(f"Epoch {epoch+1:03d}/{args.epochs} - Train Loss: {epoch_train_loss:.4f} - Valid Loss: {current_valid_loss:.4f}")
        
        model_save_path = os.path.join(log_dir, f"model_{log_name}.pth")
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test)
    logging.info(f"Final Test Loss: {test_loss.item():.4f}")

    # ==========================
    # 绘图部分 (替代原有的 JSON)
    # ==========================
    plt.figure(figsize=(10, 6))
    
    # 绘制训练和验证损失曲线
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['valid_loss'], label='Valid Loss', color='orange', linewidth=2)
    
    # 图表装饰
    plt.title(f'Training and Validation Loss\n({log_name})', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像到对应日志目录
    plot_path = os.path.join(log_dir, f"loss_curve_{log_name}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    logging.info(f"Loss curve saved to: {plot_path}")
    
    # 直接在屏幕上显示图像
    plt.show()

if __name__ == "__main__":
    main()