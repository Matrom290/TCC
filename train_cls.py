import argparse
import os
from pathlib import Path

# Desativar WandB para evitar problemas de login
os.environ["WANDB_DISABLED"] = "true"

# Definir argumentos do script
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="MarmoreTurco.yaml", help="Arquivo YAML do dataset")
parser.add_argument("--hyp", type=str, default="hyp.scratch-high.yaml", help="Arquivo YAML de hiperparâmetros")
parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch")
parser.add_argument("--total-batch-size", type=int, default=64, help="Tamanho total do batch para DDP")
parser.add_argument("--img-size", type=int, default=224, help="Tamanho da imagem")
parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado")
parser.add_argument("--device", default="cuda:0", help="Dispositivo de treinamento")
parser.add_argument("--workers", type=int, default=8, help="Número de processos para carregar os dados")
parser.add_argument("--project", default="runs/train_cls", help="Diretório para salvar os resultados")
parser.add_argument("--name", default="exp", help="Nome da pasta de saída")
parser.add_argument("--rank", type=int, default=-1, help="Rank para treinamento distribuído")
parser.add_argument("--local-rank", type=int, default=-1, help="Rank do nó local para DDP")

# Criar objeto `opt`
opt = parser.parse_args()
opt.save_dir = Path(opt.project) / opt.name

# Criar diretório de salvamento
Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

# Agora podemos importar os pacotes que dependem de `opt`
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from models.yolo import Model  # Importa o YOLO modificado para classificação
from utils.general import increment_path, set_logging, check_img_size
from utils.torch_utils import select_device, ModelEMA
from utils.dataloaders import create_classification_dataloader
from utils.loggers import GenericLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Função principal de treinamento
def train(opt):
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    import logging

    # Criar um logger válido
    console_logger = logging.getLogger("train_cls")
    console_logger.setLevel(logging.INFO)

    # Corrigir a chamada de GenericLogger
    logger = GenericLogger(opt=opt, console_logger=console_logger) if opt.rank in {-1, 0} else None


    # Configuração do treinamento distribuído
    if opt.local_rank != -1:
        dist.init_process_group("nccl", init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
    opt.world_size = 1 if opt.local_rank == -1 else dist.get_world_size()
    opt.batch_size = opt.total_batch_size // opt.world_size
    
    # Carregar hiperparâmetros do arquivo `hyp.yaml`
    with open(opt.hyp, "r") as f:
        hyp = yaml.safe_load(f)

    # Carregar o YAML do dataset
    with open(opt.data, "r") as f:
        data_yaml = yaml.safe_load(f)
    root_dir = Path(data_yaml["path"])
    train_loader = create_classification_dataloader(path=root_dir / data_yaml["train"],
                                                imgsz=opt.img_size,
                                                batch_size=opt.batch_size,
                                                augment=True,
                                                workers=opt.workers)

    val_loader = create_classification_dataloader(path=root_dir / data_yaml["val"],
                                                imgsz=opt.img_size,
                                                batch_size=opt.batch_size,
                                                augment=False,
                                                workers=opt.workers)

    test_loader = create_classification_dataloader(path=root_dir / data_yaml["test"],
                                                imgsz=opt.img_size,
                                                batch_size=opt.batch_size,
                                                augment=False,
                                                workers=opt.workers) if "test" in data_yaml else None

    # Criar modelo YOLO para classificação
    model = Model(nc=len(data_yaml["names"])).to(device).half()
    if opt.local_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    ema = ModelEMA(model) if opt.rank in {-1, 0} else None
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.epochs, pct_start=0.3)

    # Loop de Treinamento
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.half())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(train_loader))
        
        # Executar validação após cada época
        os.system(f"python val.py --data {opt.data} --weights {save_dir}/best_model.pth")

    # Salvar modelo treinado
    torch.save(ema.ema.state_dict() if ema else model.state_dict(), save_dir / "best_model.pth")
    print("Modelo salvo com sucesso!")

train(opt)
