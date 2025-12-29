import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist


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


def generate_static_labels(min_distances, thresholds):
    w_dist, d_dist = thresholds
    labels = np.zeros(len(min_distances), dtype=int)
    for i in range(len(min_distances)):
        if min_distances[i] <= d_dist:
            labels[i] = 2
        elif min_distances[i] <= w_dist:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def generate_dynamic_labels(min_distances, thresholds, prediction_horizon=15.0):
    w_dist, d_dist = thresholds
    num_frames = len(min_distances)
    labels = np.zeros(num_frames, dtype=int)
    v = np.zeros(num_frames)
    v[1:] = min_distances[1:] - min_distances[:-1]
    proj_dists = min_distances + (v * prediction_horizon)
    for i in range(num_frames):
        d_eff = max(0, proj_dists[i])
        if d_eff <= d_dist:
            labels[i] = 2
        elif d_eff <= w_dist:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def process_and_save_labels(dataset_path, output_root, thresholds, dyn_horizon):
    static_root = os.path.join(output_root, "static")
    dynamic_root = os.path.join(output_root, "dynamic")

    if not os.path.exists(dataset_path):
        print(f"ERRORE: La cartella {dataset_path} non esiste.")
        return

    for sample_name in os.listdir(dataset_path):
        sample_path = os.path.join(dataset_path, sample_name)
        if os.path.isdir(sample_path):
            os.makedirs(os.path.join(static_root, sample_name), exist_ok=True)
            os.makedirs(os.path.join(dynamic_root, sample_name), exist_ok=True)

            for filename in os.listdir(sample_path):
                if filename.endswith(".pkl"):
                    f_path = os.path.join(sample_path, filename)
                    h_pts, r_pts = extract_points(f_path)
                    if h_pts is not None and r_pts is not None:
                        distances = calculate_min_distances(h_pts, r_pts)
                        s_lbl = generate_static_labels(distances, thresholds)
                        d_lbl = generate_dynamic_labels(distances, thresholds, dyn_horizon)

                        out_name = f"{os.path.splitext(filename)[0]}_labels.npy"
                        np.save(os.path.join(static_root, sample_name, out_name), s_lbl)
                        np.save(os.path.join(dynamic_root, sample_name, out_name), d_lbl)
                        print(f"Saved: {sample_name}/{out_name}")


if __name__ == "__main__":
    # Calcolo percorsi assoluti basati sulla posizione di questo file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    LABELS_DIR = os.path.join(BASE_DIR, "labels")

    THRESHOLDS = (300.0, 150.0)
    HORIZON = 15.0

    process_and_save_labels(DATASET_DIR, LABELS_DIR, THRESHOLDS, HORIZON)