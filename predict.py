import torch
import os
import glob
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集类
class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self, normal_dir, pneumonia_dir, transform=None):
        self.normal_images = glob.glob(os.path.join(normal_dir, "*.jpeg"))
        self.pneumonia_images = glob.glob(os.path.join(pneumonia_dir, "*.jpeg"))
        self.image_paths = self.normal_images + self.pneumonia_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.pneumonia_images)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 使用PIL读取图片
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义数据预处理（ResNet要求的输入尺寸和标准化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet通常要求224x224的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet预训练的标准化
])

# 加载测试集
normal_dir = 'chest_xray/test/normal'
pneumonia_dir = 'chest_xray/test/pneumonia'
dataset = PneumoniaDataset(normal_dir, pneumonia_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 加载ResNet模型并调整最后的全连接层
model = models.resnet18(pretrained=True)  # 你也可以选择其他ResNet版本，如resnet50
model.fc = nn.Linear(model.fc.in_features, 2)  # 输出2个类别：normal和pneumonia
model = model.to(device)

# 加载模型权重
model.load_state_dict(torch.load('resnet18_chest_xray.pth', map_location=device))

# 设置模型为评估模式
model.eval()

# 用于保存评价指标的列表
accuracies = []
precisions = []
recalls = []
f1_scores = []

# 用于保存预测结果的路径
save_dir = 'predictions_results'
os.makedirs(save_dir, exist_ok=True)

# 开始进行预测和评价
all_preds = []
all_labels = []

with torch.no_grad():  # 关闭梯度计算
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 获取模型预测
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # 计算评价指标
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # 保存每个图像的预测结果
        for i in range(len(images)):
            img_path = dataset.image_paths[i + len(all_labels)]  # 调整索引
            pred = preds[i].item()
            pred_image = transforms.ToPILImage()(images[i].cpu())
            save_res_path = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_pred_{pred}.jpeg')
            pred_image.save(save_res_path)  # 保存预测图像

        # 累加所有的标签和预测结果用于最终计算
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算并打印平均评价指标
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

# 绘制评价指标曲线
epochs = np.arange(1, len(accuracies) + 1)

plt.figure(figsize=(10, 6))

# 绘制准确率
plt.subplot(2, 2, 1)
plt.plot(epochs, accuracies, label="Accuracy", color='b')
plt.xlabel('Test Image Index')
plt.ylabel('Accuracy')
plt.title('Accuracy per Image')
plt.grid(True)

# 绘制精确度
plt.subplot(2, 2, 2)
plt.plot(epochs, precisions, label="Precision", color='g')
plt.xlabel('Test Image Index')
plt.ylabel('Precision')
plt.title('Precision per Image')
plt.grid(True)

# 绘制召回率
plt.subplot(2, 2, 3)
plt.plot(epochs, recalls, label="Recall", color='r')
plt.xlabel('Test Image Index')
plt.ylabel('Recall')
plt.title('Recall per Image')
plt.grid(True)

# 绘制F1分数
plt.subplot(2, 2, 4)
plt.plot(epochs, f1_scores, label="F1 Score", color='c')
plt.xlabel('Test Image Index')
plt.ylabel('F1 Score')
plt.title('F1 Score per Image')
plt.grid(True)

# 调整布局
plt.tight_layout()
# 显示图形
plt.show()
