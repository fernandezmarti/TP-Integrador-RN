import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import io
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch.nn.init as init

def plot_to_tensorboard(fig, writer, tag, step):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1) / 255.0
    writer.add_image(tag, image, global_step=step)
    plt.close(fig)


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Función para matriz de confusión y clasificación

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

class MLPClassifier(nn.Module):
        def __init__(self, input_size=64*64*3, dropout = 0.0, num_classes=10, init_type=None, BatchNorm= False ):
            super().__init__()
            self.init_type= init_type
            if BatchNorm== True:
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            else:
                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 512),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            self.init_weights()

        def forward(self, x):
            return self.model(x)
        
        def init_weights(self):
            if self.init_type is not None:
                for m in self.model:
                    if isinstance(m, nn.Linear):
                        if self.init_type == 'He':
                            init.kaiming_normal_(m.weight, nonlinearity='relu')
                        elif self.init_type == 'xavier':
                            init.xavier_uniform_(m.weight)
                        elif self.init_type == 'uniform':
                            init.uniform_(m.weight,a=0,b=1 )
                        nn.init.zeros_(m.bias)


class CNNClassifier(nn.Module):
    def __init__(self, input_size, dropout = 0.0, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3, padding = 1, padding_mode = "reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3, padding = 1, padding_mode = "reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear((input_size//4)**2*32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

