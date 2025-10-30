import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold,train_test_split
import pdb
import numpy as np

class MMCDataset(Dataset):
    def __init__(self, x, x_length, y):
        self.x = torch.from_numpy(x)
        self.x_length = torch.from_numpy(x_length)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.x_length[idx], self.y[idx]

class ContrastiveMMCDataset(Dataset):
    def __init__(self, x, x_length, y, queue_size=4096, momentum=0.999, feature_dim=500):
        """
        初始化对比学习数据集
        Args:
            x: 输入数据
            x_length: 序列长度
            y: 标签 (one-hot编码)
            queue_size: 特征队列大小
            momentum: 动量系数
            feature_dim: 特征维度
        """
        self.x = torch.from_numpy(x)
        self.x_length = torch.from_numpy(x_length)
        self.y = torch.from_numpy(y)
        
        # 动量编码器相关参数
        self.momentum = momentum
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        
        # 构建正负样本索引
        self.pos_pairs, self.neg_pairs = self._build_sample_pairs()
        
        # # 初始化特征队列
        # self.register_queue()
        
        # 初始化队列指针
        self.queue_ptr = 0

    def _build_sample_pairs(self):
        """
        构建正负样本对
        Returns:
            pos_pairs: List[tuple], 正样本对的索引
            neg_pairs: List[tuple], 负样本对的索引
        """
        num_samples = len(self.y)
        pos_pairs = []
        neg_pairs = []
        
        # 获取每个样本的类别
        labels = torch.argmax(self.y, dim=1).numpy()
        
        # 为每个样本构建正负样本对
        for anchor_idx in range(num_samples):
            anchor_label = labels[anchor_idx]
            
            # 找到相同类别的样本作为正样本
            pos_indices = np.where(labels == anchor_label)[0]
            pos_indices = pos_indices[pos_indices != anchor_idx]  # 排除自身
            
            # 找到不同类别的样本作为负样本
            neg_indices = np.where(labels != anchor_label)[0]
            
            # 随机选择一个正样本和一个负样本
            if len(pos_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                pos_pairs.append((anchor_idx, pos_idx))
            
            if len(neg_indices) > 0:
                neg_idx = np.random.choice(neg_indices)
                neg_pairs.append((anchor_idx, neg_idx))
        
        return pos_pairs, neg_pairs

    def __getitem__(self, idx):
        """
        返回锚点样本及其对应的正负样本
        """
        # 获取锚点样本
        anchor_x = self.x[idx]
        anchor_length = self.x_length[idx]
        anchor_y = self.y[idx]
        
        # 获取正样本
        pos_idx = None
        pos_x = None
        pos_length = None
        for pair in self.pos_pairs:
            if pair[0] == idx:
                pos_idx = pair[1]
                pos_x = self.x[pos_idx]
                pos_length = self.x_length[pos_idx]
                break
        
        # 获取负样本
        neg_idx = None
        neg_x = None
        neg_length = None
        for pair in self.neg_pairs:
            if pair[0] == idx:
                neg_idx = pair[1]
                neg_x = self.x[neg_idx]
                neg_length = self.x_length[neg_idx]
                break
        
        # 如果没有找到正样本，使用自身
        if pos_x is None:
            pos_x = anchor_x
            pos_length = anchor_length
            pos_idx = idx
            
        # 如果没有找到负样本，随机选择一个不同类别的样本
        if neg_x is None:
            anchor_label = torch.argmax(anchor_y)
            other_indices = [i for i in range(len(self.y)) 
                           if torch.argmax(self.y[i]) != anchor_label]
            if other_indices:
                neg_idx = np.random.choice(other_indices)
                neg_x = self.x[neg_idx]
                neg_length = self.x_length[neg_idx]
            else:
                # 如果实在没有负样本，使用自身但加入噪声
                neg_x = anchor_x + torch.randn_like(anchor_x) * 0.1
                neg_length = anchor_length
                neg_idx = idx

        return {
            'anchor': {
                'x': anchor_x,
                'x_length': anchor_length,
                'y': anchor_y,
                'index': idx
            },
            'positive': {
                'x': pos_x,
                'x_length': pos_length,
                'index': pos_idx
            },
            'negative': {
                'x': neg_x,
                'x_length': neg_length,
                'index': neg_idx
            }
        }
    
    def __len__(self):
        return self.x.shape[0]

    # ... 其他方法保持不变 ...

