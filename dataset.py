import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据集路径
data_dir = 'chest_xray'

# 数据预处理（转换为张量，归一化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 针对ImageNet预训练模型的标准化
])

# 使用ImageFolder加载数据
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# 使用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印数据集的大小
print(f"train: {len(train_dataset)}")
print(f"val: {len(val_dataset)}")
print(f"test: {len(test_dataset)}")