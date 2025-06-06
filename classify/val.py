# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_img_size, check_requirements, colorstr,
                           increment_path, print_args)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    data=ROOT / '../datasets/mnist',  # dataset dir
    weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / 'runs/val-cls',  # save to project/name
    name='exp',  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
    save_plots=False,   # <-- nova flag
    save_images=False,  # <-- nova flag
):
    # ─── Definir save_dir sempre, quer seja chamada direta ou via train.py ───
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    # ─────────────────────────────────────────────────────────────────────────    
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        half &= device.type != 'cpu'
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)
        half = model.fp16
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Dataloader
        data = Path(data)
        test_dir = data / 'test' if (data / 'test').exists() else data / 'val'
        dataloader = create_classification_dataloader(
            path=test_dir,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=False,
            rank=-1,
            workers=workers
        )

    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(), Profile(), Profile())
    n = len(dataloader)
    action = 'validating' if dataloader.dataset.root.stem == 'val' else 'testing'
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)
    with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)
            with dt[1]:
                y = model(images)
            with dt[2]:
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if criterion:
                    loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"

    if verbose:
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in model.names.items():
            aci = acc[targets == i]
            top1i, top5i = aci.mean(0).tolist()
            LOGGER.info(f"{c:>24}{aci.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")
        t = tuple(x.t / len(dataloader.dataset.samples) * 1E3 for x in dt)
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # ------- Adições para salvar plots e imagens se solicitado -------
        # ------- salvar matriz de confusão se solicitado -------
    if save_plots:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # 1) Montar arrays de verdadeiros e predições
        y_true = targets.cpu().numpy()
        y_pred = pred[:, 0].cpu().numpy()

        # 2) Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # 3) Plotar
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = [model.names[i] for i in range(len(model.names))]
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label', xlabel='Predicted label',
               title='Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # annotate counts
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(save_dir / 'confusion_matrix.png')
        plt.close(fig)
        
        # ──── Cálculo e salvamento de precision/recall/F1 ────
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

        # 4) Calcula métricas macro
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro  = recall_score   (y_true, y_pred, average='macro', zero_division=0)
        f1_macro   = f1_score       (y_true, y_pred, average='macro', zero_division=0)

        # 5) Gera relatório detalhado por classe
        report = classification_report(
            y_true,
            y_pred,
            target_names=[model.names[i] for i in sorted(model.names)],
            zero_division=0
        )

        # 6) Salva num arquivo de texto
        metrics_file = save_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write(f"Precision (macro): {prec_macro:.4f}\n")
            f.write(f"Recall    (macro): {rec_macro:.4f}\n")
            f.write(f"F1-score  (macro): {f1_macro:.4f}\n\n")
            f.write("=== Classification Report ===\n")
            f.write(report)   
            
    # ──── Recalcular scores para curvas P/R/F1 vs threshold (com interpolação) ────
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    # 1) Prepara listas
    all_true = []
    all_scores = []

    # 2) Coleta probabilidades em todo o testloader
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type!='cpu'):
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)                            # [B, C]
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_scores.append(probs)
            all_true.append(lbls.numpy())
    all_scores = np.vstack(all_scores)   # (N, C)
    all_true   = np.hstack(all_true)     # (N,)

    # 3) Calcula curvas por classe
    precisions = []
    recalls    = []
    thresholds_list = []
    for c in range(all_scores.shape[1]):
        p, r, t = precision_recall_curve((all_true==c).astype(int), all_scores[:, c])
        precisions.append(p)
        recalls.append(r)
        thresholds_list.append(t)

    # 4) Define um grid comum de threshold
    t_grid = np.linspace(0, 1, 100)  # 100 pontos de 0 a 1
    p_interp = []
    r_interp = []
    f1_interp = []

    def f1_from_pr(p, r):
        return 2 * (p * r) / (p + r + 1e-20)

    # 5) Interpola cada curva nesse grid
    for p, r, t in zip(precisions, recalls, thresholds_list):
        p_vals = np.interp(t_grid, t, p[:-1])   # descarta último valor
        r_vals = np.interp(t_grid, t, r[:-1])
        f1_vals = f1_from_pr(p_vals, r_vals)
        p_interp.append(p_vals)
        r_interp.append(r_vals)
        f1_interp.append(f1_vals)

    # 6) Calcula média macro no grid
    p_avg = np.mean(p_interp, axis=0)
    r_avg = np.mean(r_interp, axis=0)
    f1_avg = np.mean(f1_interp, axis=0)

    # 7) Agora plota cada curva
    import matplotlib.pyplot as plt
    for name, y_vals, ylabel, fname in [
        ('Precision', p_avg, 'Precision', 'P_curve.png'),
        ('Recall',    r_avg, 'Recall',    'R_curve.png'),
        ('F1-score',  f1_avg,'F1-score',  'F1_curve.png')
    ]:
        fig, ax = plt.subplots()
        ax.plot(t_grid, y_vals)
        ax.set(xlabel='Threshold', ylabel=ylabel, title=f'{name} vs Threshold')
        fig.savefig(save_dir / fname)
        plt.close(fig)

    # 8) Precision–Recall Curve (macro) no grid
    fig, ax = plt.subplots()
    ax.plot(r_avg, p_avg)
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall Curve (macro)')
    fig.savefig(save_dir / 'PR_curve.png')
    plt.close(fig)

    # 6) Plot e salva cada curva separadamente
    import matplotlib.pyplot as plt

    # Precision vs Threshold
    fig, ax = plt.subplots()
    ax.plot(t_grid, p_avg)
    ax.set(xlabel='Threshold', ylabel='Precision', title='Precision vs Threshold')
    fig.savefig(save_dir / 'P_curve.png')
    plt.close(fig)

    # Recall vs Threshold
    fig, ax = plt.subplots()
    ax.plot(t_grid, r_avg)
    ax.set(xlabel='Threshold', ylabel='Recall', title='Recall vs Threshold')
    fig.savefig(save_dir / 'R_curve.png')
    plt.close(fig)

    # F1 vs Threshold
    fig, ax = plt.subplots()
    ax.plot(t_grid, f1_avg)
    ax.set(xlabel='Threshold', ylabel='F1-score', title='F1-score vs Threshold')
    fig.savefig(save_dir / 'F1_curve.png')
    plt.close(fig)

    # Precision–Recall Curve (macro)
    fig, ax = plt.subplots()
    ax.plot(r_avg, p_avg)
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall Curve (macro)')
    fig.savefig(save_dir / 'PR_curve.png')
    plt.close(fig)
    # -----------------------------------------------------------
 # ──────── Bloco de salvamento de imagens ────────
    if save_images:
        from utils.plots import imshow_cls

        # 1) Pega um batch de teste
        images, labels = next(iter(dataloader))
        logits = model(images.to(device))
        preds = logits.argmax(1)

        # 2) Converte labels e preds para listas de ints
        labels_list = labels.cpu().tolist()    # [0, 3, 1, …]
        preds_list  = preds.cpu().tolist()     # [0, 3, 1, …]

        # 3) Chama imshow_cls só com val.py alterado
        imshow_cls(
            images,           # im
            labels_list,      # labels → lista de ints
            preds_list,       # pred   → lista de ints
            model.names,      # names
            f=save_dir / 'test_images.jpg'
        )
    # -------------------------------------------------------------

    return top1, top5, loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../datasets/mnist', help='dataset path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', nargs='?', const=True, default=True, help='verbose output')
    parser.add_argument('--project', default=ROOT / 'runs/val-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--save-plots', action='store_true', help='save confusion matrix and PR curve as .png')
    parser.add_argument('--save-images', action='store_true', help='save grid of test images with predictions')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
