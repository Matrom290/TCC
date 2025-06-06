# train_cls_dual.py atualizado e otimizado

import os
import sys
import yaml
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from models.yolo import ClassifyYOLO
from utils.dataloaders import create_classification_dataloader
from utils.loggers import GenericLogger
from utils.callbacks import EarlyStopping

# Configura seed para reprodutibilidade
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False

def main(opt):
    # Pasta de salvamento
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = GenericLogger(opt=opt, console_logger=True)

    # Dados
    with open(opt.data, 'r') as f:
        data_yaml = yaml.safe_load(f)

    train_path = Path(data_yaml['train'])
    val_path = Path(data_yaml['val'])
    test_path = Path(data_yaml['test'])
    nc = len(data_yaml['names'])

    # Model
    model = ClassifyYOLO(nc=nc).to(opt.device)
    
    # Otimizador e scheduler
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # Dataloaders
    train_loader = create_classification_dataloader(train_path, imgsz=opt.img_size, batch_size=opt.batch_size, augment=True, workers=opt.workers)
    val_loader = create_classification_dataloader(val_path, imgsz=opt.img_size, batch_size=opt.batch_size, augment=False, workers=opt.workers)
    test_loader = create_classification_dataloader(test_path, imgsz=opt.img_size, batch_size=opt.batch_size, augment=False, workers=opt.workers)

    # Early Stopping
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Loop de treino
    best_acc = 0
    train_losses, val_losses = [], []
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        preds, labels = [], []

        pbar = tqdm(train_loader, desc=f"Treinando Epoch {epoch+1}/{opt.epochs}")
        for images, targets in pbar:
            images, targets = images.to(opt.device), targets.to(opt.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(labels, preds)
        train_losses.append(train_loss)
        accuracy_list.append(train_acc)

        # Validação
        model.eval()
        running_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(opt.device), targets.to(opt.device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets)
                running_val_loss += loss.item()

                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        precision_list.append(val_prec)
        recall_list.append(val_recall)
        f1_list.append(val_f1)

        # Logs
        print(f"\nEpoch {epoch+1}/{opt.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        # Early Stopping
        early_stopping(val_loss, model, save_dir)
        if early_stopping.early_stop:
            print("\nEarly stopping acionado!")
            break

        scheduler.step()

    # Salvar último modelo
    torch.save(model.state_dict(), save_dir / "last_model.pth")

    # Teste
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(opt.device), targets.to(opt.device)
            outputs = model(images)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(targets.cpu().numpy())

    # Relatório final
    print("\n🔹 Relatório Final 🔹")
    print(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Test Precision: {precision_score(test_labels, test_preds, average='macro', zero_division=0):.4f}")
    print(f"Test Recall: {recall_score(test_labels, test_preds, average='macro', zero_division=0):.4f}")
    print(f"Test F1-Score: {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}")
    print("\n", classification_report(test_labels, test_preds))

    # Matriz de Confusão
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(save_dir / "confusion_matrix.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path para dataset YAML')
    parser.add_argument('--epochs', type=int, default=100, help='Total de épocas')
    parser.add_argument('--batch-size', type=int, default=16, help='Tamanho do batch')
    parser.add_argument('--img-size', type=int, default=224, help='Tamanho da imagem')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--workers', type=int, default=8, help='Num workers')
    parser.add_argument('--project', type=str, default='runs/train-cls', help='Diretório para salvar')
    parser.add_argument('--name', type=str, default='exp', help='Nome da experiência')
    opt = parser.parse_args()

    main(opt)
