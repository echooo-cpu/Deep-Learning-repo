import yaml
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """加载并预处理数据"""
    # 加载糖尿病数据集
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化（均值为0，方差为1）- 神经网络必须做
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32).reshape(-1, 1)  # [N, 1] 形状
    X_test = torch.tensor(X_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32).reshape(-1, 1)
    
    print(f"train dataset size: {X_train.shape} test dataset size: {X_test.shape}")
    return X_train, y_train, X_test, y_test

class FNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置读取参数
        input_dim = config['model']['input_dim']
        hidden_dim = config['model']['hidden_dim']
        output_dim = config['model']['output_dim']
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
def main():
    # 1. 加载配置
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("配置:", config)
    
    # 2. 加载数据（使用你自己的函数）
    X_train, y_train, X_test, y_test = load_data()
    
    # 3. 创建模型（传入配置）
    model = FNN(config)
    
    # 4. 训练（从配置读取参数）
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 训练
        model.train()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每10轮打印
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test)
            print(f"Epoch {epoch+1}, 训练损失: {loss.item():.2f}, 测试损失: {test_loss.item():.2f}")


if __name__ == "__main__":
    main()