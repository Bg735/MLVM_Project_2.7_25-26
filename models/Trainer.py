import copy

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import optim, nn

from scripts.generate_report import save_and_print_report


class Trainer:
    def __init__(self,
                 training_params,
                 checkpoint_path,
                 model: nn.Module,
                 data_loaders: dict,
                 device,
                 loss_criterion,
                 optimizer_cls=optim.Adam,
                 optimizer_kwargs=None
                 ):

        self.device = device
        print(f"Trainer initialized on device: {self.device}")
        self.model = model.to(self.device)
        self.loaders = data_loaders
        self.criterion = loss_criterion.to(self.device)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.training_params = training_params
        self.checkpoint_path = checkpoint_path

    def train(self):
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.training_params.learning_rate, **self.optimizer_kwargs)

        if self.training_params.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        else:
            scheduler = None

        if self.training_params.load_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path, self.device))
            print('Loaded model checkpoint.')

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.training_params.num_epochs):
            self.model.train()
            t_loss, t_total = 0.0, 0
            train_preds, train_labels = [], []

            for inputs, labels in self.loaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                t_total += labels.size(0)

                train_preds.extend(pred.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            epoch_t_loss = t_loss / t_total
            t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
                train_labels, train_preds, average=None, labels=[0, 1, 2], zero_division=0
            )

            # --- VALIDATION ---
            self.model.eval()
            v_loss, v_total = 0.0, 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for inputs, labels in self.loaders['validation']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    v_loss += loss.item() * inputs.size(0)
                    _, pred = torch.max(outputs, 1)
                    v_total += labels.size(0)

                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            epoch_v_loss = v_loss / v_total
            v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average=None, labels=[0, 1, 2], zero_division=0
            )

            history['train_loss'].append(epoch_t_loss)
            history['train_precision'].append(t_prec)
            history['train_recall'].append(t_rec)
            history['train_f1'].append(t_f1)

            history['val_loss'].append(epoch_v_loss)
            history['val_precision'].append(v_prec)
            history['val_recall'].append(v_rec)
            history['val_f1'].append(v_f1)

            print(f"\nEpoch {epoch + 1}/{self.training_params.num_epochs}")
            print(f"  [Train] Loss: {epoch_t_loss:.4f}")
            print(f"  [Val]   Loss: {epoch_v_loss:.4f}")
            print("  --- Class details (Train | Val) ---")
            print(f"SAFE     | Prec: {t_prec[0]:.2f}|{v_prec[0]:.2f} Rec: {t_rec[0]:.2f}|{v_rec[0]:.2f} F1: {t_f1[0]:.2f}|{v_f1[0]:.2f}")
            print(f"WARNING  | Prec: {t_prec[1]:.2f}|{v_prec[1]:.2f} Rec: {t_rec[1]:.2f}|{v_rec[1]:.2f} F1: {t_f1[1]:.2f}|{v_f1[1]:.2f}")
            print(f"CRITICAL | Prec: {t_prec[2]:.2f}|{v_prec[2]:.2f} Rec: {t_rec[2]:.2f}|{v_rec[2]:.2f} F1: {t_f1[2]:.2f}|{v_f1[2]:.2f}")
            print("-" * 40)

            if self.training_params.use_scheduler:
                scheduler.step(epoch_v_loss)

            if epoch % self.training_params.checkpoint_interval == self.training_params.checkpoint_interval - 1:
                if epoch_v_loss < best_val_loss:
                    best_val_loss = epoch_v_loss
                    patience_counter = 0

                    torch.save(self.model.state_dict(), env.checkpoint_path)
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print(f"    -> New best model saved.")

                else:
                    patience_counter += 1
                    if patience_counter >= self.training_params.patience:
                        print(f"\nEarly Stopping activated: negligible progress after {self.training_params.patience*self.training_params.checkpoint_interval} epochs.")
                        break

        self.model.load_state_dict(best_model_wts)
        self.evaluate_test()

        print("--- HISTORY ---")

        print(f"{'Epoch':<6} | {'Tr.Loss':<8} | {'Val.Loss':<8} | {'Prec. (S, W, C)':<18} | {'Rec. (S, W, C)':<18} | {'F1 (S, W, C)':<18}")
        print("-" * 95)

        for epoch in range(len(history['train_loss'])):
            tl = history['train_loss'][epoch]
            vl = history['val_loss'][epoch]

            p_str = ", ".join([f"{v:.2f}" for v in history['val_precision'][epoch]])
            r_str = ", ".join([f"{v:.2f}" for v in history['val_recall'][epoch]])
            f_str = ", ".join([f"{v:.2f}" for v in history['val_f1'][epoch]])

            print(f"{epoch:<6} | {tl:<8.3f} | {vl:<8.3f} | ({p_str}) | ({r_str}) | ({f_str})")

        return history

    def evaluate_test(self):
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.loaders['test']:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)

                    probs = torch.softmax(outputs, dim=1)

                    preds = []
                    for p in probs:
                        preds.append(torch.argmax(p).item())

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            save_and_print_report(
                cfg, seed, all_preds, all_labels
            )

