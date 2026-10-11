import yaml
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import LearningRateScheduler
from src.dataset import prepare_dataset
from src.model_lmu import build_lmu_model
from src.model_lstm import build_lstm_model


def _classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
    }


def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Running experiment: {config['experiment']['name']}")

    # Reproducibility
    seed = config["experiment"].get("seed_list", [42])[0]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. Prepare data
    print("Preparing dataset...")
    train_data, val_data, test_data, df_meta = prepare_dataset(config)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    # Add channel dim expected by model
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Plot class distribution
    os.makedirs("results/figures/dataset", exist_ok=True)
    plt.figure()
    plt.hist(y_train, bins=np.arange(4) - 0.5, rwidth=0.8)
    plt.title("Class Distribution (Train)")
    plt.savefig("results/figures/dataset/class_distribution.png")
    plt.close()

    lr           = config["training"]["learning_rate"]
    epochs       = config["training"]["epochs"]
    batch        = config["training"]["batch_size"]
    decay_factor = config["training"]["lr_decay_factor"]
    decay_epochs = config["training"]["lr_decay_epochs"]

    def lr_schedule(epoch, current_lr):
        if epoch > 0 and epoch % decay_epochs == 0:
            return current_lr * decay_factor
        return current_lr

    # ======================== LMU ========================
    print("Building LMU model...")
    lmu = build_lmu_model(config)
    lmu.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    lmu.summary()

    print("Training LMU model...")
    os.makedirs("results/figures/lmu", exist_ok=True)
    lmu_history = lmu.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=[LearningRateScheduler(lr_schedule)],
        verbose=2,
    )

    lmu_eval = lmu.evaluate(X_test, y_test, verbose=0)
    lmu_prob = lmu.predict(X_test, verbose=0)
    lmu_labels = np.argmax(lmu_prob, axis=1)
    lmu_metrics = _classification_metrics(y_test, lmu_labels)
    print(f"LMU  Test loss={lmu_eval[0]:.4f}  acc={lmu_eval[1]:.4f}")

    plt.figure()
    plt.plot(lmu_history.history["loss"], label="train")
    plt.plot(lmu_history.history["val_loss"], label="val")
    plt.title("LMU Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("results/figures/lmu/loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(lmu_history.history["accuracy"], label="train")
    plt.plot(lmu_history.history["val_accuracy"], label="val")
    plt.title("LMU Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig("results/figures/lmu/accuracy_curve.png"); plt.close()

    # ======================== LSTM ========================
    print("Building LSTM model...")
    lstm = build_lstm_model(config)
    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    lstm.summary()

    print("Training LSTM model...")
    os.makedirs("results/figures/lstm", exist_ok=True)
    lstm_history = lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=[LearningRateScheduler(lr_schedule)],
        verbose=2,
    )

    lstm_eval = lstm.evaluate(X_test, y_test, verbose=0)
    lstm_prob = lstm.predict(X_test, verbose=0)
    lstm_labels = np.argmax(lstm_prob, axis=1)
    lstm_metrics = _classification_metrics(y_test, lstm_labels)
    print(f"LSTM Test loss={lstm_eval[0]:.4f}  acc={lstm_eval[1]:.4f}")

    plt.figure()
    plt.plot(lstm_history.history["loss"], label="train")
    plt.plot(lstm_history.history["val_loss"], label="val")
    plt.title("LSTM Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("results/figures/lstm/loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(lstm_history.history["accuracy"], label="train")
    plt.plot(lstm_history.history["val_accuracy"], label="val")
    plt.title("LSTM Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig("results/figures/lstm/accuracy_curve.png"); plt.close()

    # ======================== Save results ========================
    os.makedirs("results", exist_ok=True)
    with open("results/model_comparison.csv", "w") as f:
        f.write("Model,our_total_params,paper_total_params,accuracy,macro_precision,macro_recall,macro_f1,notes\n")
        f.write(
            f"LMU,{lmu.count_params()},41131,{lmu_metrics['accuracy']:.4f},"
            f"{lmu_metrics['precision']:.4f},{lmu_metrics['recall']:.4f},"
            f"{lmu_metrics['f1']:.4f},Hazan-inspired theta=1.0\n"
        )
        f.write(
            f"LSTM,{lstm.count_params()},40899,{lstm_metrics['accuracy']:.4f},"
            f"{lstm_metrics['precision']:.4f},{lstm_metrics['recall']:.4f},"
            f"{lstm_metrics['f1']:.4f},Baseline LSTM comparison\n"
        )

    summary = {
        "seed": seed,
        "lmu_test_accuracy":  float(lmu_eval[1]),
        "lmu_test_loss":      float(lmu_eval[0]),
        "lmu_metrics":        lmu_metrics,
        "lstm_test_accuracy": float(lstm_eval[1]),
        "lstm_test_loss":     float(lstm_eval[0]),
        "lstm_metrics":       lstm_metrics,
        "lmu_params":         int(lmu.count_params()),
        "lstm_params":        int(lstm.count_params()),
        "train_samples":      int(X_train.shape[0]),
        "val_samples":        int(X_val.shape[0]),
        "test_samples":       int(X_test.shape[0]),
    }
    with open("results/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaving test predictions for audit...")
    np.savez("results/test_predictions.npz", y_true=y_test, lmu_preds=lmu_prob, lstm_preds=lstm_prob)

    print("\nAll done. Results saved in results/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_experiment("configs/baseline.yaml")
