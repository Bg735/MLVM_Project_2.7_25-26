"""
lead_time_analysis.py
=====================
Analisi dell'Anticipo Temporale ("Time-to-Critical-Event") — Sezione 4.3

LOGICA CORRETTA
---------------
L'analisi NON viene fatta sull'array piatto del DataLoader (dove i campioni
di soggetti diversi sono concatenati e la continuità temporale è persa).

Viene fatta FILE PER FILE: per ogni .pkl nel test set, si ricostruisce
l'intera sequenza temporale originale, si fa inferenza frame per frame,
e si misura l'anticipo all'interno di quella sequenza continua.

HOW TO RUN
----------
    python scripts/lead_time_analysis.py \
        --checkpoint  checkpoints/dynamic_best.pt \
        --dataset     dataset/ \
        --window_size 10 \
        --frame_ms    40 \
        --train_split 0.7 \
        --val_split   0.15 \
        --output_dir  reports/

Parametri:
  --checkpoint   path al file .pt del modello LSTM salvato durante il training
  --dataset      path alla cartella dataset/
  --window_size  finestra temporale usata durante il training (default 10)
  --frame_ms     periodo di campionamento in ms (Tc nel tuo codice = 40 ms)
  --train_split  stessa frazione usata nel notebook (default 0.7)
  --val_split    stessa frazione usata nel notebook (default 0.15)
  --output_dir   cartella dove salvare il PDF
  --hidden_dim   hidden size LSTM (default 128)
  --num_layers   numero layer LSTM (default 2)
  --seed         seed usato nel training (per riproducibilità dello split)
"""

import argparse
import glob
import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Argomenti
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--dataset",     required=True)
    p.add_argument("--window_size", type=int,   default=10)
    p.add_argument("--frame_ms",    type=float, default=40.0)
    p.add_argument("--train_split", type=float, default=0.7)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--output_dir",  default="../reports")
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Modello LSTM (identico a DynamicModel.py)
# ---------------------------------------------------------------------------

class DynamicSafetyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# Stesso split subject-stratificato del notebook
# ---------------------------------------------------------------------------

def get_test_subjects(dataset_path, train_split, val_split):
    """
    Replica esatta della logica di split del notebook.
    Ritorna la lista dei subject assegnati al test set.
    """
    samples_dir = os.path.join(dataset_path, "samples")
    all_subjects = sorted([
        d for d in os.listdir(samples_dir)
        if os.path.isdir(os.path.join(samples_dir, d))
    ])

    # Conta i frame CRITICAL per ogni subject (usando le label dynamic già generate)
    subject_counts = {}
    for subj in all_subjects:
        label_path = os.path.join(dataset_path, "labels", "dynamic", subj)
        npy_files  = glob.glob(os.path.join(label_path, "*.npy"))
        all_lbls   = np.concatenate([np.load(f) for f in npy_files]) if npy_files else np.array([])
        subject_counts[subj] = np.bincount(all_lbls.astype(int), minlength=3)

    sorted_subjs = sorted(all_subjects, key=lambda s: subject_counts[s][2], reverse=True)

    train_subjs, val_subjs, test_subjs = [], [], []
    counts_train = np.zeros(3)
    counts_val   = np.zeros(3)
    counts_test  = np.zeros(3)
    test_split   = 1.0 - train_split - val_split

    for subj in sorted_subjs:
        c = subject_counts[subj]
        tot = counts_train[2] + counts_val[2] + counts_test[2] + c[2]
        if tot == 0:
            tot = 1

        diff_train = (counts_train[2] + c[2]) / tot - train_split
        diff_val   = (counts_val[2]   + c[2]) / tot - val_split
        diff_test  = (counts_test[2]  + c[2]) / tot - test_split

        choice = np.argmin([diff_train, diff_val, diff_test])
        if choice == 0:
            train_subjs.append(subj); counts_train += c
        elif choice == 1:
            val_subjs.append(subj);   counts_val   += c
        else:
            test_subjs.append(subj);  counts_test  += c

    print(f"\n  Train subjects : {train_subjs}")
    print(f"  Val subjects   : {val_subjs}")
    print(f"  Test subjects  : {test_subjs}")
    return train_subjs, val_subjs, test_subjs


# ---------------------------------------------------------------------------
# Calcola media/std sul training set (identico al notebook)
# ---------------------------------------------------------------------------

def compute_train_normalization(dataset_path, train_subjects, window_size):
    """
    Ricostruisce le feature di distanza per tutti i file dei train_subjects
    e calcola media/std per la normalizzazione — identico al notebook.
    """
    all_frames = []
    for subj in train_subjects:
        subj_path = os.path.join(dataset_path, "samples", subj)
        for pkl_f in sorted(glob.glob(os.path.join(subj_path, "*.pkl"))):
            with open(pkl_f, "rb") as f:
                raw = pickle.load(f)
            human_seq = np.array([frame[0] for frame in raw])
            robot_seq = np.array([frame[1] for frame in raw])
            for i in range(len(human_seq)):
                d = cdist(human_seq[i], robot_seq[i], metric="euclidean")
                all_frames.append(d.flatten())

    all_frames = np.array(all_frames, dtype=np.float32)
    mean = np.mean(all_frames, axis=0)
    std  = np.std(all_frames,  axis=0)
    print(f"  Normalizzazione calcolata su {len(all_frames)} frame di training")
    return mean, std


# ---------------------------------------------------------------------------
# CORE: inferenza e analisi anticipo su un singolo file .pkl
# ---------------------------------------------------------------------------

def analyze_single_file(pkl_path, label_path, model, mean, std,
                         window_size, frame_ms, device):
    """
    Dato un singolo file .pkl (una sequenza temporale continua):
      1. Ricostruisce le feature di distanza frame per frame
      2. Carica le label ground truth
      3. Fa inferenza con finestre scorrevoli (identico al DataLoader)
      4. Per ogni evento CRITICAL nel ground truth, cerca il primo
         WARNING/CRITICAL predetto PRIMA di quell'evento nella stessa sequenza

    Ritorna una lista di dict, uno per ogni evento CRITICAL trovato.
    """
    # Carica dati grezzi
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    human_seq = np.array([frame[0] for frame in raw], dtype=np.float32)
    robot_seq = np.array([frame[1] for frame in raw], dtype=np.float32)
    num_frames = human_seq.shape[0]

    # Costruisce la matrice di distanze (N_frames, N_features)
    n_human = human_seq.shape[1]
    n_robot = robot_seq.shape[1]
    dists = np.zeros((num_frames, n_human * n_robot), dtype=np.float32)
    for i in range(num_frames):
        d = cdist(human_seq[i], robot_seq[i], metric="euclidean")
        dists[i] = d.flatten()

    # Carica label ground truth
    file_stem  = os.path.splitext(os.path.basename(pkl_path))[0]
    label_file = os.path.join(label_path, f"{file_stem}_labels.npy")
    if not os.path.exists(label_file):
        return []
    y_true = np.load(label_file).astype(int)

    # Normalizzazione (identica al notebook: (x - mean) / std)
    std_safe = np.where(std < 1e-7, 1e-7, std)
    dists_norm = (dists - mean) / std_safe

    # Costruisce le finestre scorrevoli ed esegue inferenza frame per frame
    # Ogni finestra è identica a quella costruita nel CollisionDataset
    mean_t = torch.tensor(mean,     dtype=torch.float32)
    std_t  = torch.tensor(std_safe, dtype=torch.float32)

    model.eval()
    y_pred = np.zeros(num_frames, dtype=int)

    with torch.no_grad():
        # Processo a batch per efficienza
        batch_windows = []
        for i in range(num_frames):
            start = i - window_size + 1
            if start < 0:
                real   = dists_norm[0: i + 1]
                pad    = np.tile(dists_norm[0], (window_size - len(real), 1))
                window = np.concatenate((pad, real), axis=0)
            else:
                window = dists_norm[start: i + 1]
            batch_windows.append(window)

        # Inferenza in un unico batch (efficiente)
        x = torch.tensor(np.array(batch_windows), dtype=torch.float32).to(device)
        outputs = model(x)
        y_pred  = torch.argmax(outputs, dim=1).cpu().numpy().astype(int)

    # Analisi anticipo temporale su questa sequenza continua
    results = []
    i = 0
    while i < num_frames:
        if y_true[i] == 2:  # inizio evento CRITICAL nel ground truth
            t_critical = i

            # Scorri all'indietro nella STESSA sequenza
            found = False
            for j in range(t_critical - 1, -1, -1):
                if y_true[j] == 2:
                    break  # era già un altro evento CRITICAL
                if y_pred[j] in (1, 2):  # modello ha predetto WARNING o CRITICAL
                    lead_frames = t_critical - j
                    results.append({
                        "file":             os.path.basename(pkl_path),
                        "t_critical_frame": t_critical,
                        "t_signal_frame":   j,
                        "lead_frames":      lead_frames,
                        "lead_ms":          lead_frames * frame_ms,
                        "signal_class":     int(y_pred[j]),  # 1=WARNING, 2=CRITICAL
                        "detected":         True,
                    })
                    found = True
                    break

            if not found:
                results.append({
                    "file":             os.path.basename(pkl_path),
                    "t_critical_frame": t_critical,
                    "t_signal_frame":   None,
                    "lead_frames":      None,
                    "lead_ms":          None,
                    "signal_class":     None,
                    "detected":         False,
                })

            # Avanza oltre il blocco CRITICAL corrente
            while i < num_frames and y_true[i] == 2:
                i += 1
        else:
            i += 1

    return results


# ---------------------------------------------------------------------------
# Grafici e report
# ---------------------------------------------------------------------------

def make_plots(all_results, frame_ms, output_dir, plot_cap_ms=1500):
    """
    plot_cap_ms: soglia oltre la quale i valori vengono esclusi DAI GRAFICI
                 (le statistiche in tabella usano sempre tutti i dati).
    """
    os.makedirs(output_dir, exist_ok=True)

    detected     = [r for r in all_results if r["detected"]]
    missed_count = sum(1 for r in all_results if not r["detected"])
    total        = len(all_results)
    covered      = len(detected)

    # Array completo (usato per le statistiche in tabella)
    lead_times_full = np.array([r["lead_ms"] for r in detected]) if detected else np.array([])
    lt_warn_full    = np.array([r["lead_ms"] for r in detected if r["signal_class"] == 1])
    lt_crit_full    = np.array([r["lead_ms"] for r in detected if r["signal_class"] == 2])

    # Array filtrato (usato solo per i grafici)
    outliers        = int(np.sum(lead_times_full > plot_cap_ms)) if len(lead_times_full) else 0
    lead_times_plot = lead_times_full[lead_times_full <= plot_cap_ms] if len(lead_times_full) else np.array([])
    lt_warn_plot    = lt_warn_full[lt_warn_full   <= plot_cap_ms] if len(lt_warn_full)  else np.array([])
    lt_crit_plot    = lt_crit_full[lt_crit_full   <= plot_cap_ms] if len(lt_crit_full)  else np.array([])

    coverage_pct = 100.0 * covered / total if total > 0 else 0.0

    # Stampa a console (statistiche complete, senza filtro)
    print("\n" + "=" * 60)
    print("  ANALISI ANTICIPO TEMPORALE — Time-to-Critical-Event")
    print("=" * 60)
    print(f"  Totale eventi CRITICAL nel ground truth : {total}")
    print(f"  Rilevati con anticipo                   : {covered}  ({coverage_pct:.1f}%)")
    print(f"  Non rilevati (missed)                   : {missed_count}")
    if len(lead_times_full) > 0:
        print(f"  Anticipo [ms]  min    : {np.min(lead_times_full):.0f}")
        print(f"  Anticipo [ms]  mediana: {np.median(lead_times_full):.0f}")
        print(f"  Anticipo [ms]  media  : {np.mean(lead_times_full):.0f}")
        print(f"  Anticipo [ms]  max    : {np.max(lead_times_full):.0f}")
        print(f"  Anticipo [ms]  std    : {np.std(lead_times_full):.0f}")
        if len(lt_warn_full) > 0:
            print(f"  → segnale WARNING   (n={len(lt_warn_full):3d}): media {np.mean(lt_warn_full):.0f} ms")
        if len(lt_crit_full) > 0:
            print(f"  → segnale CRITICAL  (n={len(lt_crit_full):3d}): media {np.mean(lt_crit_full):.0f} ms")
        if outliers > 0:
            print(f"  [Grafici] {outliers} outlier esclusi dalla visualizzazione (> {plot_cap_ms} ms)")
    print("=" * 60)

    # --- Figura ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf   = os.path.join(output_dir, f"lead_time_analysis_{timestamp}.pdf")

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Istogramma principale (dati filtrati <= plot_cap_ms)
    ax_hist = fig.add_subplot(gs[0, :])
    if len(lead_times_plot) > 0:
        bins = np.arange(0, plot_cap_ms + frame_ms * 2, frame_ms)
        if len(lt_warn_plot) > 0:
            ax_hist.hist(lt_warn_plot,  bins=bins, color="steelblue", alpha=0.75,
                         label=f"Primo segnale = WARNING (n={len(lt_warn_plot)})",
                         edgecolor="white", linewidth=0.4)
        if len(lt_crit_plot) > 0:
            ax_hist.hist(lt_crit_plot, bins=bins, color="firebrick", alpha=0.65,
                         label=f"Primo segnale = CRITICAL predetto (n={len(lt_crit_plot)})",
                         edgecolor="white", linewidth=0.4)
        # Le linee media/mediana usano i dati COMPLETI (più onesti)
        ax_hist.axvline(np.mean(lead_times_full),   color="black",  lw=2, linestyle="--",
                        label=f"Media: {np.mean(lead_times_full):.0f} ms")
        ax_hist.axvline(np.median(lead_times_full), color="orange", lw=2, linestyle=":",
                        label=f"Mediana: {np.median(lead_times_full):.0f} ms")
        ax_hist.legend(fontsize=10)
        ax_hist.set_xlim(0, plot_cap_ms)
    outlier_note = f" — {outliers} outlier > {plot_cap_ms} ms esclusi dal grafico" if outliers > 0 else ""
    ax_hist.set_title(
        f"Distribuzione dell'Anticipo Temporale\n"
        "Distanza temporale tra il primo segnale predetto e l'evento CRITICAL reale",
        fontsize=12, fontweight="bold"
    )
    ax_hist.set_xlabel("Anticipo [ms]", fontsize=12)
    ax_hist.set_ylabel("Numero di eventi CRITICAL", fontsize=12)
    ax_hist.grid(axis="y", alpha=0.35)

    # 2. Tabella riassuntiva (statistiche COMPLETE, senza filtro)
    ax_tab = fig.add_subplot(gs[1, 0])
    ax_tab.axis("off")
    rows = [
        ["Totale eventi CRITICAL",        str(total)],
        ["Rilevati in anticipo",           f"{covered}  ({coverage_pct:.1f}%)"],
        ["Non rilevati (missed)",          str(missed_count)],
        ["Anticipo min [ms]",              f"{np.min(lead_times_full):.0f}"    if len(lead_times_full) else "—"],
        ["Anticipo mediana [ms]",          f"{np.median(lead_times_full):.0f}" if len(lead_times_full) else "—"],
        ["Anticipo medio [ms]",            f"{np.mean(lead_times_full):.0f}"   if len(lead_times_full) else "—"],
        ["Anticipo max [ms]",              f"{np.max(lead_times_full):.0f}"    if len(lead_times_full) else "—"],
        ["Std dev [ms]",                   f"{np.std(lead_times_full):.0f}"    if len(lead_times_full) else "—"],
        ["Media (segnale WARNING) [ms]",   f"{np.mean(lt_warn_full):.0f}"      if len(lt_warn_full)    else "—"],
        ["Media (segnale CRITICAL) [ms]",  f"{np.mean(lt_crit_full):.0f}"      if len(lt_crit_full)    else "—"],
        [f"Outlier esclusi dal grafico",   f"{outliers}  (> {plot_cap_ms} ms)"],
    ]
    tbl = ax_tab.table(cellText=rows, colLabels=["Metrica", "Valore"],
                       cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c5f8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf3fb")
        # Evidenzia riga outlier in giallo tenue
        if row == len(rows):
            cell.set_facecolor("#fff8dc")
    ax_tab.set_title("Tabella Riassuntiva",
                     fontsize=11, fontweight="bold", pad=8)

    # 3. Box-plot filtrato (<= plot_cap_ms)
    ax_box = fig.add_subplot(gs[1, 1])
    box_data   = []
    box_labels = []
    if len(lt_warn_plot) > 0:
        box_data.append(lt_warn_plot)
        box_labels.append(f"WARNING\n(n={len(lt_warn_plot)})")
    if len(lt_crit_plot) > 0:
        box_data.append(lt_crit_plot)
        box_labels.append(f"CRITICAL pred.\n(n={len(lt_crit_plot)})")
    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.45,
                            medianprops=dict(color="orange", linewidth=2),
                            flierprops=dict(marker="o", markersize=3,
                                            markerfacecolor="gray", alpha=0.5))
        colors = ["steelblue", "firebrick"]
        for patch, color in zip(bp["boxes"], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_xticks(range(1, len(box_labels) + 1))
        ax_box.set_xticklabels(box_labels, fontsize=10)
        ax_box.set_ylabel("Anticipo [ms]", fontsize=11)
        ax_box.set_ylim(0, plot_cap_ms)
        ax_box.grid(axis="y", alpha=0.35)
    ax_box.set_title(f"Box-plot per tipo di segnale",
                     fontsize=11, fontweight="bold")

    plt.suptitle("LSTM — Analisi Predittiva: Time-to-Critical-Event",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\n  PDF salvato in: {out_pdf}")
    return out_pdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Window size: {args.window_size} frame | Frame period: {args.frame_ms} ms")

    # 1. Determina quali subject vanno nel test set (stessa logica del notebook)
    print("\n[STEP 1] Determinazione split soggetti...")
    train_subjs, val_subjs, test_subjs = get_test_subjects(
        args.dataset, args.train_split, args.val_split
    )

    # 2. Calcola normalizzazione sui train subject
    print("\n[STEP 2] Calcolo normalizzazione dal training set...")
    mean, std = compute_train_normalization(args.dataset, train_subjs, args.window_size)

    # 3. Carica modello
    print(f"\n[STEP 3] Caricamento modello da {args.checkpoint}...")
    # Determina input_dim dal primo file disponibile
    first_subj = test_subjs[0]
    first_pkl  = sorted(glob.glob(
        os.path.join(args.dataset, "samples", first_subj, "*.pkl")
    ))[0]
    with open(first_pkl, "rb") as f:
        raw = pickle.load(f)
    h0 = np.array(raw[0][0])
    r0 = np.array(raw[0][1])
    input_dim = h0.shape[0] * r0.shape[0]
    print(f"  input_dim rilevato: {input_dim}")

    model = DynamicSafetyClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=3,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"  Parametri modello: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Analisi file per file sui test subject
    print(f"\n[STEP 4] Analisi sequenza per sequenza sui {len(test_subjs)} test subject...")
    all_results = []

    for subj in test_subjs:
        subj_sample_path = os.path.join(args.dataset, "samples", subj)
        subj_label_path  = os.path.join(args.dataset, "labels", "dynamic", subj)
        pkl_files        = sorted(glob.glob(os.path.join(subj_sample_path, "*.pkl")))

        print(f"\n  Subject [{subj}] — {len(pkl_files)} sequenze")
        for pkl_f in pkl_files:
            results = analyze_single_file(
                pkl_path=pkl_f,
                label_path=subj_label_path,
                model=model,
                mean=mean,
                std=std,
                window_size=args.window_size,
                frame_ms=args.frame_ms,
                device=device,
            )
            n_crit = sum(1 for r in results if r is not None)
            n_det  = sum(1 for r in results if r.get("detected"))
            print(f"    {os.path.basename(pkl_f)}: {n_crit} eventi CRITICAL, {n_det} rilevati in anticipo")
            all_results.extend(results)

    # 5. Grafici
    if not all_results:
        print("\n[WARN] Nessun evento CRITICAL trovato. Controlla dataset e split.")
        return

    print(f"\n[STEP 5] Generazione grafici...")
    make_plots(all_results, args.frame_ms, args.output_dir)

    # Salva risultati grezzi
    raw_path = os.path.join(args.output_dir,
                            f"lead_time_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
    np.save(raw_path, all_results)
    print(f"  Risultati grezzi salvati in: {raw_path}")
    print("\n[DONE]")


def compute_train_normalization(dataset_path, train_subjects, window_size):
    all_frames = []
    for subj in train_subjects:
        subj_path = os.path.join(dataset_path, "samples", subj)
        for pkl_f in sorted(glob.glob(os.path.join(subj_path, "*.pkl"))):
            with open(pkl_f, "rb") as f:
                raw = pickle.load(f)
            human_seq = np.array([frame[0] for frame in raw])
            robot_seq = np.array([frame[1] for frame in raw])
            for i in range(len(human_seq)):
                d = cdist(human_seq[i], robot_seq[i], metric="euclidean")
                all_frames.append(d.flatten())
    all_frames = np.array(all_frames, dtype=np.float32)
    mean = np.mean(all_frames, axis=0)
    std  = np.std(all_frames,  axis=0)
    print(f"  Normalizzazione calcolata su {len(all_frames)} frame di training")
    return mean, std


if __name__ == "__main__":
    main()


    '''
    "/Users/mauriziomelillo/POLITO/machine learning/PROGETTO/mlprogetto/.venv/bin/python" \
  "/Users/mauriziomelillo/POLITO/machine learning/PROGETTO/mlprogetto/scripts/lead_time_analysis.py" \
  --checkpoint "/Users/mauriziomelillo/POLITO/machine learning/PROGETTO/mlprogetto/models/best_model_dyn.pkl" \
  --dataset    "/Users/mauriziomelillo/POLITO/machine learning/PROGETTO/mlprogetto/dataset" \
  --window_size 10 \
  --frame_ms 40 \
  --output_dir "/Users/mauriziomelillo/POLITO/machine learning/PROGETTO/mlprogetto/reports"

    '''