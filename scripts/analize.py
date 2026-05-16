import os
import glob
import numpy as np

# 1. Definisci la funzione
def analyze_dataset_stats(root_dir):
    print("--- LABEL ANALYSIS ---")
    label_dirs = ['static', 'dynamic']

    for l_type in label_dirs:
        path = os.path.join(root_dir, 'labels', l_type)
        if not os.path.exists(path):
            raise OSError(f"Folder {path} not found.")

        all_labels = []
        npy_files = glob.glob(os.path.join(path, '**/*.npy'), recursive=True)

        print(f"Reading {len(npy_files)} label files for {l_type}...")
        for f in npy_files:
            lbls = np.load(f)
            all_labels.extend(lbls)

        all_labels = np.array(all_labels)
        if len(all_labels) == 0:
            raise OSError("No label found.") # Attenzione: qui nel codice originale mancava "raise"

        unique, counts = np.unique(all_labels, return_counts=True)
        total = sum(counts)
        print(f"\nStats {l_type.upper()}:")
        for val, count in zip(unique, counts):
            name = "SAFE" if val == 0 else "WARNING" if val == 1 else "CRITICAL"
            print(f"  Class {val} ({name}): {count} ({count / total * 100:.2f}%)")

# 2. Esegui lo script
if __name__ == "__main__":
    # Assicurati che questo percorso punti correttamente alla cartella "dataset"
    # Se il tuo file check_stats.py è dentro la cartella "scripts", il path è:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    
    # QUESTO È IL PASSAGGIO CHIAVE: chiama effettivamente la funzione!
    analyze_dataset_stats(DATASET_DIR)