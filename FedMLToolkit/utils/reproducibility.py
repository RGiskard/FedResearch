import torch
import random
import numpy as np

def set_seeds(seed=42):
    """Fijar semillas para garantizar reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Operaciones determinísticas en GPU
    torch.backends.cudnn.benchmark = False     # Desactiva optimizaciones dinámicas para reproducibilidad
