import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime
import os

def plot_learning_curves(history, output_dir="../reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    best_epoch = np.argmin(history['val_loss'])
    max_epoch = len(history['train_loss']) - 1
    
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=best_epoch, color='black', linestyle=':', linewidth=2, label=f'Best Model (Ep {best_epoch})')
    
    ax1.axvspan(best_epoch, max_epoch, color='yellow', alpha=0.2, label='Zona Overfitting')
    
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    
    val_prec_crit = [p[2] for p in history['val_precision']]
    val_rec_crit = [r[2] for r in history['val_recall']]
    
    ax2.plot(val_prec_crit, label='Val Precision (CRITICAL)', color='forestgreen', linewidth=2)
    ax2.plot(val_rec_crit, label='Val Recall (CRITICAL)', color='purple', linewidth=2, linestyle='--')
    ax2.axvline(x=best_epoch, color='black', linestyle=':', linewidth=2, label='Best Model')
    
    ax2.axvspan(best_epoch, max_epoch, color='yellow', alpha=0.2, label='Degrado Recall')
    
    ax2.set_title('Prestazioni Classe CRITICAL nel Tempo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"learning_curves_{timestamp}.pdf")
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_distance_distribution(dataset, output_dir="../reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    min_distances = []
    
    for x in dataset.data:
        min_d = np.min(x) 
        min_distances.append(min_d)
        
    plt.figure(figsize=(10, 6))
    
    sns.histplot(min_distances, bins=100, binrange=(0, 1000), color='steelblue', stat='density')
    
    plt.axvline(x=130, color='red', linestyle='--', linewidth=2, label='Soglia CRITICAL (130mm)')
    plt.axvline(x=350, color='orange', linestyle='--', linewidth=2, label='Soglia WARNING (350mm)')
    
    plt.axvspan(0, 130, alpha=0.1, color='red')
    plt.axvspan(130, 350, alpha=0.1, color='orange')
    plt.axvspan(350, 1000, alpha=0.1, color='green')
    
    plt.title('Distribuzione Fisica delle Distanze Minime nel Dataset', fontsize=15, fontweight='bold')
    plt.xlabel('Distanza Minima Umano-Robot (mm)', fontsize=13)
    plt.ylabel('Densità (Frequenza)', fontsize=13)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"distance_distribution_{timestamp}.pdf")
    plt.savefig(output_path, dpi=300)
    plt.show()