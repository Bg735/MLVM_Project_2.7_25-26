import os
from datetime import datetime

import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

REPORTS_PATH="../reports"

def save_and_print_report(cfg, seed, all_labels, all_preds):
    CLASSES = ['SAFE', 'WARNING', 'CRITICAL']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPORTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"report_{timestamp}.pdf")

    report_str = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)

    print(f"\n--- TEST REPORT ---")
    print(report_str)

    config_content = yaml.dump(cfg.model_dump(), default_flow_style=False)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    fig_full = plt.figure(figsize=(12, 18))
    gs = fig_full.add_gridspec(3, 1, height_ratios=[1, 1, 2])


    ax0 = fig_full.add_subplot(gs[0]); ax0.axis('off')
    ax0.text(1.0, 0.95, f"Current seed: {seed}", fontsize=9, family='monospace', verticalalignment='top', horizontalalignment='right', transform=ax0.transAxes)
    ax0.text(0, 0.95, f"--- CONFIGURATION ---\n\n{config_content}", fontsize=9, family='monospace', verticalalignment='top')

    ax1 = fig_full.add_subplot(gs[1]); ax1.axis('off')
    ax1.text(0, 0.95, f"--- TEST REPORT ---\n\n{report_str}", fontsize=10, family='monospace', verticalalignment='top')

    ax2 = fig_full.add_subplot(gs[2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax2)
    ax2.set_title("Confusion Matrix")

    ax2.set_xlabel('Predicted Labels', fontweight='bold')
    ax2.set_ylabel('True Labels', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig_full)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix")

    plt.xlabel('Predicted Labels', fontweight='bold')
    plt.ylabel('True Labels', fontweight='bold')

    plt.show()