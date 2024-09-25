import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# Model definition
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

# Transform function
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Dataset class
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

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim=0)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Test function
def test_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            scores = torch.sigmoid(outputs)
            predicted = (scores > 0.5).float()
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    metrics = calculate_metrics(all_labels, all_predictions)

    return metrics, all_predictions, all_labels, all_scores

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = torch.zeros(2, 2)
    cm[0, 0] = torch.sum((1 - y_true) * (1 - y_pred))  # TN
    cm[0, 1] = torch.sum((1 - y_true) * y_pred)        # FP
    cm[1, 0] = torch.sum(y_true * (1 - y_pred))        # FN
    cm[1, 1] = torch.sum(y_true * y_pred)              # TP

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_per_class_f1(f1_scores, save_path):
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(f1_scores)), f1_scores)
    plt.title('F1 Score per Class')
    plt.xlabel('Class Index')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_path):
    precision, recall, _ = precision_recall_curve(y_true.numpy().ravel(), y_scores.numpy().ravel())
    average_precision = average_precision_score(y_true.numpy(), y_scores.numpy(), average="micro")

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:0.2f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_heatmap(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(y_pred[:50, :].T, cmap='YlOrRd', vmin=0, vmax=1)
    plt.title('Model Predictions Heatmap (First 50 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Class Index')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_sample_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        # Display image
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Sample {i+1}')

        # Display true labels and predictions
        axes[i, 1].barh(range(91), labels[i], label='True', alpha=0.5)
        axes[i, 1].barh(range(91), predictions[i], label='Predicted', alpha=0.5)
        axes[i, 1].set_yticks([])
        axes[i, 1].set_xlabel('Label Presence')
        axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    test_dataset = datasets.CocoDetection(
        root='/Users/vinayak/Developement/VS Code/PythonML/coco2017/val2017',
        annFile='/Users/vinayak/Developement/VS Code/PythonML/coco2017/annotations/instances_val2017.json',
        transform=get_transform()
    )
    test_dataset = COCOMultiLabelDataset(torch.utils.data.Subset(test_dataset, range(1000)))  # Use a subset for faster testing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize and load the trained model
    num_classes = 91
    model = COCOCNN(num_classes)
    model.load_state_dict(torch.load('coco_cnn_model.pth', map_location=device))
    model.to(device)

    # Test the model
    metrics, all_predictions, all_labels, all_scores = test_model(model, test_loader, device)

    # Print and save results
    print("Test Results:")
    print(f"Accuracy: {metrics['accuracy'].mean().item():.4f}")
    print(f"Precision: {metrics['precision'].mean().item():.4f}")
    print(f"Recall: {metrics['recall'].mean().item():.4f}")
    print(f"F1 Score: {metrics['f1'].mean().item():.4f}")

    # Save metrics to a JSON file
    with open('test_results.json', 'w') as f:
        json.dump({k: v.tolist() for k, v in metrics.items()}, f, indent=4)

    # Generate and save visualizations
    plot_confusion_matrix(all_labels.flatten(), all_predictions.flatten(), 'confusion_matrix.png')
    plot_per_class_f1(metrics['f1'], 'per_class_f1.png')
    plot_precision_recall_curve(all_labels, all_scores, 'precision_recall_curve.png')
    plot_prediction_heatmap(all_labels, all_predictions, 'prediction_heatmap.png')
    visualize_sample_predictions(model, test_loader, device)

    print("Results and visualizations saved.")

if __name__ == '__main__':
    main()