import glob
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist

STATIC_THRESHOLDS = (350, 130)
DYNAMIC_THRESHOLDS = (500, 300)

def extract_points(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        h_seq, r_seq = [], []
        for frame in data:
            h_seq.append(frame[0])
            r_seq.append(frame[1])
        return np.array(h_seq), np.array(r_seq)
    except Exception as e:
        print(f"Error extracting {pkl_path}: {e}")
        return None, None


def calculate_min_distances(human_pts, robot_pts):
    num_frames = human_pts.shape[0]
    min_dists = np.zeros(num_frames)
    for i in range(num_frames):
        dists = cdist(human_pts[i], robot_pts[i], metric='euclidean')
        min_dists[i] = np.min(dists)
    return min_dists


def generate_static_labels(min_distances):
    condlist = [
        min_distances <= STATIC_THRESHOLDS[1],
        min_distances <= STATIC_THRESHOLDS[0]
    ]
    choicelist = [2, 1]
    return np.select(condlist, choicelist, default=0)

def static_label(min_distances):
    if min_distances <= STATIC_THRESHOLDS[1]:
        label = 2
    elif min_distances <= STATIC_THRESHOLDS[0]:
        label = 1
    else:
        label = 0
    return label

def generate_dynamic_labels(min_distances):
    """
    Usa il TTC (Time to Collision) per generare label predittive.
    TTC = Distanza / Velocità di avvicinamento.
    """
    num_frames = len(min_distances)
    labels = np.zeros(num_frames, dtype=int)

    # Calcolo della velocità (variazione di distanza: d_t - d_{t-1})
    v = np.zeros(num_frames)
    v[1:] = min_distances[1:] - min_distances[:-1]

    for i in range(num_frames):

        if v[i] >= 0:
            ttc = float('inf')
        else:
            ttc = min_distances[i] / abs(v[i])

        if ttc <= DYNAMIC_THRESHOLDS[1] or min_distances[i] <= STATIC_THRESHOLDS[1]:
            labels[i] = 2  # CRITICAL
        elif ttc <= DYNAMIC_THRESHOLDS[0] or min_distances[i] <= STATIC_THRESHOLDS[0]:
            labels[i] = 1  # WARNING
        else:
            labels[i] = 0  # SAFE

        if ttc == float('inf') and labels[i] != 0:
            labels[i] = labels[i]-1

    return labels


def process_and_save_labels(samples_path, labels_path):
    static_root = os.path.join(labels_path, "static")
    dynamic_root = os.path.join(labels_path, "dynamic")

    for sample_name in os.listdir(samples_path):
        sample_path = os.path.join(samples_path, sample_name)
        if os.path.isdir(sample_path):
            os.makedirs(os.path.join(static_root, sample_name), exist_ok=True)
            os.makedirs(os.path.join(dynamic_root, sample_name), exist_ok=True)

            for filename in os.listdir(sample_path):
                if filename.endswith(".pkl"):
                    f_path = os.path.join(sample_path, filename)
                    h_pts, r_pts = extract_points(f_path)
                    if h_pts is not None and r_pts is not None:
                        distances = calculate_min_distances(h_pts, r_pts)
                        s_lbl = generate_static_labels(distances)
                        d_lbl = generate_dynamic_labels(distances)

                        out_name = f"{os.path.splitext(filename)[0]}_labels.npy"
                        np.save(os.path.join(static_root, sample_name, out_name), s_lbl)
                        np.save(os.path.join(dynamic_root, sample_name, out_name), d_lbl)
                        print(f"Saved: {sample_name}/{out_name}")


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
            OSError("No label found.")

        unique, counts = np.unique(all_labels, return_counts=True)
        total = sum(counts)
        print(f"\nStats {l_type.upper()}:")
        for val, count in zip(unique, counts):
            name = "SAFE" if val == 0 else "WARNING" if val == 1 else "CRITICAL"
            print(f"  Class {val} ({name}): {count} ({count / total * 100:.2f}%)")


def analyze_distances(root_dir):
    print("\n--- DISTANCE ANALYSIS ---")
    samples_dir = os.path.join(root_dir, 'samples')

    all_min_distances = []

    pkl_files = glob.glob(os.path.join(samples_dir, '**/*.pkl'), recursive=True)
    print(f"Found {len(pkl_files)} .pkl files. Computing minimum distances...")

    for i, pkl_f in enumerate(pkl_files):
        if i % 10 == 0: print(f"\rProcessed  {i}/{len(pkl_files)}...", end="")
        h_pts, r_pts = extract_points(pkl_f)
        if h_pts is None: continue

        num_frames = h_pts.shape[0]
        for f in range(num_frames):
            dists = cdist(h_pts[f], r_pts[f], metric='euclidean')
            min_d = np.min(dists)
            all_min_distances.append(min_d)
    print(f"\rProcessed {len(pkl_files)}", end="")

    all_min_distances = np.array(all_min_distances)
    print(f"\n\nAnalysis based on {len(all_min_distances)} total frames.")
    print(f"Minimum distance: {np.min(all_min_distances):.2f}")
    print(f"Average distance: {np.mean(all_min_distances):.2f}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    SAMPLES_DIR = os.path.join(DATASET_DIR, 'samples')
    LABELS_DIR = os.path.join(DATASET_DIR, 'labels')


    process_and_save_labels(SAMPLES_DIR, LABELS_DIR)

    #analyze_dataset_stats(DATASET_DIR)
    #analyze_distances(DATASET_DIR)