import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_dataset_distribution():
    # 1. 加载数据并进行标准化 
    # (降维前一定要做标准化，尤其是PCA对绝对方差非常敏感)
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA 降维 (适合捕捉全局线性结构)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # 计算PCA解释的方差比例
    explained_variance = pca.explained_variance_ratio_

    # 3. t-SNE 降维 (擅长捕捉局部非线性聚集关系)
    # 注意: t-SNE 每次运行可能会略有不同，所以可以固定 random_state
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    # 4. 开始绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    
    # 配色方案：使用 coolwarm，深蓝代表患病轻，深红代表患病重
    cmap = 'rainbow' 

    # ----- 绘制 PCA 子图 -----
    scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, alpha=0.8, edgecolor='w', s=50)
    axes[0].set_title(f"PCA (Explains {explained_variance[0]*100:.1f}% + {explained_variance[1]*100:.1f}% Variance)")
    axes[0].set_xlabel("First Principal Component (PC1)")
    axes[0].set_ylabel("Second Principal Component (PC2)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # ----- 绘制 t-SNE 子图 -----
    scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, alpha=0.8, edgecolor='w', s=50)
    axes[1].set_title("t-SNE (Local Non-linear Manifolds)")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 添加共享的颜色条 (Colorbar)
    cbar = fig.colorbar(scatter_pca, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
    cbar.set_label('Diabetes Progression Indicator (y)', rotation=270, labelpad=15, fontweight='bold')

    plt.suptitle("Diabetes Dataset 2D Distribution (Colored by Target 'y')", fontsize=16, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    visualize_dataset_distribution()