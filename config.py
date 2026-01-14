# amacımız bu sistemin içerisinde kullanacağımız kök dizinlerle alakalı ayarlamaları yapmak

from pathlib import Path
import logging 
import torch

# projenin çalıştığı dizin
ROOT = Path(__file__).parent # -->bundan sonra kullanacağımız iki ayrı sistem olck
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

# bu dizinler var mı yok mu? yoksa oluşturulsun:
for p in (CHECKPOINT_DIR, LOG_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Model/training defaults
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "vit_base_patch16_224" # hangi modeli kullanacağımız
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
    
# Logging
def init_logger(name:str = "pipeline", level=logging.INFO):
    logger=logging.getLogger(name)
    if logger.handlers:
        return logger # sistemin başladığıyla alakalı bir giriş sinyali
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler() # bütün senaryo boyunca loggingin çalışmasıyla alakalı
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger    

logger = init_logger() 