import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class COCOCNN(nn.Module):
    def __init__(self, num_classes):
        super(COCOCNN, self).__init__()
        self.features = nn.Sequential(
            self._make_cnn_layer(3, 64),
            self._make_cnn_layer(64, 128),
            self._make_cnn_layer(128, 256),
            self._make_cnn_layer(256, 512),
            self._make_cnn_layer(512, 1024),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def _make_cnn_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class COCOMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __getitem__(self, index):
        image, target = self.coco_dataset[index]
        labels = torch.zeros(91)  # 91 classes including background
        for ann in target:
            labels[ann['category_id']] = 1
        return image, labels

    def __len__(self):
        return len(self.coco_dataset)

def train_model(model, train_loader, val_loader, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_f1 = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_f1 += f1_score(labels, predicted)

        val_f1 /= len(val_loader)
        val_f1_scores.append(val_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation F1 Score: {val_f1:.4f}')

    return model, train_losses, val_f1_scores

def f1_score(y_true, y_pred):
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return torch.mean(f1)

def plot_training_progress(train_losses, val_f1_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, 'r-')
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def main():
    train_dataset = datasets.CocoDetection(root='/Users/vinayak/Developement/VS Code/PythonML/coco2017/train2017',
                                           annFile='/Users/vinayak/Developement/VS Code/PythonML/coco2017/annotations/instances_train2017.json',
                                           transform=get_transform())
    val_dataset = datasets.CocoDetection(root='/Users/vinayak/Developement/VS Code/PythonML/coco2017/val2017',
                                         annFile='/Users/vinayak/Developement/VS Code/PythonML/coco2017/annotations/instances_val2017.json',
                                         transform=get_transform())
    
    train_dataset = COCOMultiLabelDataset(torch.utils.data.Subset(train_dataset, range(1000)))
    val_dataset = COCOMultiLabelDataset(torch.utils.data.Subset(val_dataset, range(100)))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    num_classes = 91
    model = COCOCNN(num_classes)

    trained_model, train_losses, val_f1_scores = train_model(model, train_loader, val_loader)

    plot_training_progress(train_losses, val_f1_scores)

    torch.save(trained_model.state_dict(), 'coco_cnn_model.pth')

if __name__ == '__main__':
    main()