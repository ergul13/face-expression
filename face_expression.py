!pip -q install ultralytics kagglehub matplotlib pandas pyyaml

import os, time, logging, pathlib, json, random
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ultralytics import YOLO
import kagglehub as kh
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('emotion_training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    kaggle_dataset: str = "fatihkgg/affectnet-yolo-format"
    model_name: str = "yolov8m.pt"
    epochs: int = 100
    img_size: int = 640
    batch_size: int = -1
    patience: int = 20
    device: Optional[str] = None
    learning_rate: float = 0.01
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    seed: int = 42
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0

class EmotionDatasetManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.base_path = pathlib.Path.cwd()
        self.dataset_path = None
        self.data_yaml_path = None
        self.emotion_classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]
        self.supported_formats = ['*.jpg','*.jpeg','*.png','*.webp','*.bmp','*.tiff','*.tif']

    def download_dataset(self) -> pathlib.Path:
        logger.info(f"Dataset indiriliyor: {self.config.kaggle_dataset}")
        p = pathlib.Path(kh.dataset_download(self.config.kaggle_dataset))
        self.dataset_path = p
        logger.info(f"Dataset indirildi: {p}")
        return p

    def find_yaml_config(self) -> Optional[pathlib.Path]:
        if not self.dataset_path: return None
        ymls = list(self.dataset_path.rglob("*.yaml")) + list(self.dataset_path.rglob("*.yml"))
        for y in ymls:
            try:
                t = y.read_text(encoding="utf-8", errors="ignore")
                if "train:" in t and "val:" in t:
                    logger.info(f"YAML bulundu: {y}")
                    return y
            except Exception as e:
                logger.warning(f"YAML okunamadı {y}: {e}")
        return None

    def count_images_recursive(self, directory: pathlib.Path) -> int:
        total = 0
        for pattern in self.supported_formats:
            total += len(list(directory.rglob(pattern)))
        return total

    def count_labels_recursive(self, directory: pathlib.Path) -> int:
        return len(list(directory.rglob("*.txt")))

    def find_image_directories(self) -> Dict[str, pathlib.Path]:
        if not self.dataset_path: return {}
        dirs = {}
        for name in ["train","training"]:
            p = next(self.dataset_path.rglob(f"{name}/images"), None)
            if p and p.exists():
                dirs["train"] = p
                break
        for name in ["val","valid","validation"]:
            p = next(self.dataset_path.rglob(f"{name}/images"), None)
            if p and p.exists():
                dirs["val"] = p
                break
        logger.info(f"Bulunan dizinler: {dirs}")
        return dirs

    def create_data_yaml(self) -> pathlib.Path:
        yaml_config = self.find_yaml_config()
        image_dirs = self.find_image_directories()
        if yaml_config:
            with open(yaml_config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if "train" in cfg and "val" in cfg:
                tr, vl = pathlib.Path(cfg["train"]), pathlib.Path(cfg["val"])
                if not tr.exists() and "train" in image_dirs:
                    cfg["train"] = str(image_dirs["train"])
                if not vl.exists() and "val" in image_dirs:
                    cfg["val"] = str(image_dirs["val"])
            if "names" not in cfg: cfg["names"] = self.emotion_classes
            if "nc" not in cfg: cfg["nc"] = len(self.emotion_classes)
            out = self.base_path / "emotion_dataset_config.yaml"
        else:
            if "train" not in image_dirs or "val" not in image_dirs:
                raise FileNotFoundError("train/val images dizinleri bulunamadı.")
            cfg = {
                "train": str(image_dirs["train"]),
                "val": str(image_dirs["val"]),
                "nc": len(self.emotion_classes),
                "names": self.emotion_classes
            }
            out = self.base_path / "emotion_dataset_auto.yaml"
        with open(out, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        self.data_yaml_path = out
        logger.info(f"YAML yazıldı: {out}")
        return out

    def analyze_dataset(self) -> Dict:
        if not self.data_yaml_path: return {}
        with open(self.data_yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        stats = {
            "train_images": 0, "val_images": 0,
            "train_images_with_labels": 0, "val_images_with_labels": 0,
            "missing_train_labels": 0, "missing_val_labels": 0,
            "image_label_mismatch": False, "supported_formats": self.supported_formats
        }
        exts = tuple(x.lstrip('*') for x in self.supported_formats)

        def pair_stats(img_dir: pathlib.Path):
            lbl_dir = img_dir.parent / "labels"
            imgs = [p for p in img_dir.rglob('*') if p.suffix.lower() in exts]
            labeled = 0; missing = 0
            if lbl_dir.exists():
                for img in imgs:
                    stem = img.with_suffix('').name
                    if (lbl_dir / f"{stem}.txt").exists(): labeled += 1
                    else: missing += 1
            else:
                missing = len(imgs)
            return len(imgs), labeled, missing

        train_path = pathlib.Path(cfg["train"])
        if train_path.exists():
            ti, tlab, tmiss = pair_stats(train_path)
            stats.update({"train_images": ti, "train_images_with_labels": tlab, "missing_train_labels": tmiss})

        val_path = pathlib.Path(cfg["val"])
        if val_path.exists():
            vi, vlab, vmiss = pair_stats(val_path)
            stats.update({"val_images": vi, "val_images_with_labels": vlab, "missing_val_labels": vmiss})

        stats["image_label_mismatch"] = bool(stats["missing_train_labels"] or stats["missing_val_labels"])
        logger.info(f"Dataset analizi: {stats}")
        return stats

class EmotionTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.training_results = None
        self.validation_results = None
        self.experiment_name = f"emotion_yolo_{int(time.time())}"
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)

    def initialize_model(self) -> YOLO:
        logger.info(f"Model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        return self.model

    def train(self, data_yaml_path: pathlib.Path) -> Dict:
        if not self.model: self.initialize_model()
        device = self.config.device if self.config.device is not None else (0 if torch.cuda.is_available() else 'cpu')
        logger.info(f"Eğitim: device={device}, epochs={self.config.epochs}, seed={self.config.seed}")
        args = {
            'data': str(data_yaml_path),
            'epochs': self.config.epochs,
            'imgsz': self.config.img_size,
            'batch': self.config.batch_size,
            'device': device,
            'patience': self.config.patience,
            'project': 'runs/train',
            'name': self.experiment_name,
            'verbose': True,
            'deterministic': True,
            'save': True,
            'plots': True,
            'seed': self.config.seed,
            'lr0': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'hsv_h': self.config.hsv_h, 'hsv_s': self.config.hsv_s, 'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees, 'translate': self.config.translate,
            'scale': self.config.scale, 'shear': self.config.shear,
            'perspective': self.config.perspective, 'flipud': self.config.flipud,
            'fliplr': self.config.fliplr, 'mosaic': self.config.mosaic, 'mixup': self.config.mixup
        }
        self.training_results = self.model.train(**args)
        logger.info("Eğitim tamamlandı.")
        return self.training_results

    def validate(self, data_yaml_path: pathlib.Path) -> Dict:
        if not self.model: raise ValueError("Model hazır değil.")
        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info("Validasyon başlıyor...")
        args = {
            'data': str(data_yaml_path),
            'split': 'val',
            'imgsz': self.config.img_size,
            'batch': self.config.batch_size,
            'device': device,
            'project': 'runs/val',
            'name': self.experiment_name,
            'save_json': True,
            'plots': True,
            'verbose': True
        }
        self.validation_results = self.model.val(**args)
        logger.info("Validasyon tamamlandı.")
        return self.validation_results

class ResultsAnalyzer:
    def __init__(self, trainer: EmotionTrainer):
        self.trainer = trainer
        self.train_results_dir = pathlib.Path(trainer.training_results.save_dir) if trainer.training_results else None
        self.val_results_dir = pathlib.Path(trainer.validation_results.save_dir) if trainer.validation_results else None

    def extract_confusion_matrix_data(self) -> Optional[np.ndarray]:
        if self.val_results_dir:
            p = self.val_results_dir / "confusion_matrix.json"
            if p.exists():
                try:
                    with open(p, "r") as f:
                        return np.array(json.load(f))
                except Exception:
                    pass
        vr = self.trainer.validation_results
        if hasattr(vr, "confusion_matrix") and getattr(vr.confusion_matrix, "matrix", None) is not None:
            return vr.confusion_matrix.matrix
        return None

    def plot_training_metrics(self) -> None:
        if not self.train_results_dir: return
        csv = self.train_results_dir / "results.csv"
        if not csv.exists(): return
        df = pd.read_csv(csv)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10)); fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
        if {'metrics/mAP50(B)','metrics/mAP50-95(B)'}.issubset(df.columns):
            axes[0,0].plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', label='mAP@0.50', linewidth=2)
            axes[0,0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='mAP@0.50:0.95', linewidth=2)
            axes[0,0].set_title('Mean Average Precision'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        if {'metrics/precision(B)','metrics/recall(B)'}.issubset(df.columns):
            axes[0,1].plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Precision', linewidth=2)
            axes[0,1].plot(df['epoch'], df['metrics/recall(B)'], 'orange', label='Recall', linewidth=2)
            axes[0,1].set_title('Precision & Recall'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
        loss_cols = [c for c in df.columns if 'loss' in c.lower()]
        for c in loss_cols[:3]:
            axes[1,0].plot(df['epoch'], df[c], label=c.replace('train/',''), linewidth=2)
        axes[1,0].set_title('Training Losses'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
        if 'lr/pg0' in df.columns:
            axes[1,1].plot(df['epoch'], df['lr/pg0'], 'purple', linewidth=2); axes[1,1].set_title('Learning Rate'); axes[1,1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.train_results_dir / 'training_analysis.png', dpi=300, bbox_inches='tight'); plt.show()

    def analyze_class_performance(self) -> None:
        vr = self.trainer.validation_results
        if not vr or not hasattr(vr, 'box') or not hasattr(vr.box, 'maps'): return
        names = list(vr.names.values()) if hasattr(vr, 'names') else []
        maps = vr.box.maps
        if not names or len(maps) == 0: return
        df = pd.DataFrame({'Class': names, 'mAP@0.50:0.95': maps})
        if self.val_results_dir: df.to_csv(self.val_results_dir / 'class_performance.csv', index=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.bar(names, maps, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Classes'); ax1.set_ylabel('mAP@0.50:0.95'); ax1.set_title('Per-Class Performance'); ax1.tick_params(axis='x', rotation=45)
        for i,v in enumerate(maps): ax1.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        cm = self.extract_confusion_matrix_data()
        if cm is not None:
            im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.figure.colorbar(im, ax=ax2)
            ax2.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=names, yticklabels=names,
                    title="Confusion Matrix", ylabel='True', xlabel='Pred')
            th = cm.max()/2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                             color="white" if cm[i, j] > th else "black")
        else:
            ax2.text(0.5,0.5,'Confusion Matrix\nNot Available', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_xticks([]); ax2.set_yticks([])
        plt.tight_layout()
        out = self.val_results_dir or self.train_results_dir
        if out: plt.savefig(out / 'detailed_class_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def display_saved_plots(self) -> None:
        paths = {}
        if self.train_results_dir:
            for t,f in {"Training Results":"results.png","Training Curves":"train_batch0.png"}.items():
                p = self.train_results_dir / f
                if p.exists(): paths[t] = p
        if self.val_results_dir:
            for t,f in {"Confusion Matrix":"confusion_matrix.png","PR Curve":"PR_curve.png","F1 Curve":"F1_curve.png","Validation Batch":"val_batch0_pred.png"}.items():
                p = self.val_results_dir / f
                if p.exists(): paths[t] = p
        if not paths: return
        n = len(paths); cols = 2; rows = (n+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows)); axes = np.atleast_1d(axes).ravel()
        for ax,(title,p) in zip(axes, paths.items()):
            img = plt.imread(str(p)); ax.imshow(img); ax.set_title(title, fontweight='bold'); ax.axis('off')
        for ax in axes[len(paths):]: ax.axis('off')
        plt.tight_layout()
        out = self.val_results_dir or self.train_results_dir
        if out: plt.savefig(out / 'comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self) -> Dict:
        summary = {
            "experiment_info": {
                "name": self.trainer.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "seed": self.trainer.config.seed
            },
            "model_config": {
                "architecture": self.trainer.config.model_name,
                "epochs": self.trainer.config.epochs,
                "img_size": self.trainer.config.img_size,
                "batch_size": self.trainer.config.batch_size,
                "learning_rate": self.trainer.config.learning_rate,
                "patience": self.trainer.config.patience
            },
            "augmentation_config": {
                "hsv_h": self.trainer.config.hsv_h,
                "hsv_s": self.trainer.config.hsv_s,
                "hsv_v": self.trainer.config.hsv_v,
                "fliplr": self.trainer.config.fliplr,
                "mosaic": self.trainer.config.mosaic,
                "mixup": self.trainer.config.mixup
            }
        }
        if self.trainer.validation_results and hasattr(self.trainer.validation_results, 'box'):
            box = self.trainer.validation_results.box
            summary["validation_metrics"] = {
                "overall": {
                    "mAP50-95": float(getattr(box, 'map', 0)),
                    "mAP50": float(getattr(box, 'map50', 0)),
                    "precision": float(getattr(box, 'mp', 0)),
                    "recall": float(getattr(box, 'mr', 0))
                },
                "class_count": len(getattr(self.trainer.validation_results, 'names', {}))
            }
            if hasattr(box, 'maps') and hasattr(self.trainer.validation_results, 'names'):
                names = list(self.trainer.validation_results.names.values())
                summary["validation_metrics"]["per_class"] = [
                    {"class": c, "AP50-95": float(ap)} for c, ap in zip(names, box.maps)
                ]
        out_dir = self.val_results_dir or self.train_results_dir
        if out_dir:
            summary["output_paths"] = {
                "results_dir": str(out_dir),
                "best_weights": str(out_dir / "weights" / "best.pt"),
                "last_weights": str(out_dir / "weights" / "last.pt")
            }
            with open(out_dir / "experiment_summary.yaml", "w") as f:
                yaml.safe_dump(summary, f, sort_keys=False)
        logger.info("Özet rapor oluşturuldu.")
        return summary

def main():
    logger.info("=== Emotion Detection Pipeline ===")
    cfg = TrainingConfig()
    dm = EmotionDatasetManager(cfg)
    ds_path = dm.download_dataset()
    data_yaml = dm.create_data_yaml()
    stats = dm.analyze_dataset()
    print("\n" + "="*60 + "\nDATASET İSTATİSTİKLERİ\n" + "="*60)
    for k,v in stats.items(): print(f"{k}: {v}")
    print("="*60 + "\n")
    tr = EmotionTrainer(cfg)
    tr.initialize_model()
    print("Eğitim başlatılıyor...")
    tr.train(data_yaml)
    print("Validasyon başlatılıyor...")
    tr.validate(data_yaml)
    an = ResultsAnalyzer(tr)
    an.plot_training_metrics()
    an.analyze_class_performance()
    an.display_saved_plots()
    summary = an.generate_summary_report()
    print("\n" + "="*60 + "\nEĞİTİM TAMAMLANDI!\n" + "="*60)
    print(f"Experiment: {tr.experiment_name}")
    out = an.val_results_dir or an.train_results_dir
    print(f"Results: {out}")
    if "validation_metrics" in summary and "overall" in summary["validation_metrics"]:
        m = summary["validation_metrics"]["overall"]
        print(f"mAP@0.50: {m['mAP50']:.4f}")
        print(f"mAP@0.50:0.95: {m['mAP50-95']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall: {m['recall']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
