import argparse
import json
import logging
import sys
from pathlib import Path

# Automatically add 'src' directory to Python path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config

from sign_language.data import SignLanguageDataset, build_dataset_from_manifest, compute_vocabulary_hash
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.landmarks.schema import SCHEMA_VERSION
from sign_language.models import build_model
from sign_language.training.trainer import (
    compute_classification_metrics,
    evaluate_model,
)
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PipelineCollator:
    """Picklable, high-speed vectorized collate function for PyTorch multi-process DataLoader."""

    def __init__(self, sequence_length: int = 64) -> None:
        self.sequence_length = sequence_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        features, masks, labels, sample_ids = [], [], [], []
        target_len = self.sequence_length

        for item in batch:
            seq = np.asarray(item["landmarks"], dtype=np.float32)
            if seq.ndim == 1:
                seq = seq.reshape(1, -1)
            raw_mask = item.get("mask")
            mask = np.asarray(raw_mask, dtype=np.float32) if raw_mask is not None else np.ones_like(seq, dtype=np.float32)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)

            t_frames = seq.shape[0]
            num_points = seq.shape[1] // 3
            coords = seq.reshape(t_frames, num_points, 3)
            pt_valid = (mask.reshape(t_frames, num_points, 3) > 0.5).any(axis=-1)
            pt_mask = pt_valid[:, :, None]
            n_valid = pt_mask.sum(axis=1, keepdims=True)
            has_valid = (n_valid > 0)
            safe_n = np.maximum(n_valid, 1)

            # Vectorized spatial centering & scaling with strict float32
            center = np.where(has_valid, (coords * pt_mask).sum(axis=1, keepdims=True) / safe_n, np.float32(0.0)).astype(np.float32)
            centered = np.where(pt_mask, coords - center, np.float32(0.0)).astype(np.float32)
            scale = np.sqrt(np.where(has_valid, np.sum(np.square(centered), axis=(1, 2), keepdims=True) / (safe_n * np.float32(3.0)), np.float32(1.0))).astype(np.float32)
            norm_coords = np.where(scale >= 1e-6, centered / np.maximum(scale, np.float32(1e-6)), centered)
            norm_coords = np.where(pt_mask, norm_coords, np.float32(0.0)).astype(np.float32)
            norm_seq = norm_coords.reshape(t_frames, -1)

            # Vectorized PyTorch linear interpolation across time
            t_tensor = torch.from_numpy(norm_seq).permute(1, 0).unsqueeze(0).float()
            res_t = torch.nn.functional.interpolate(t_tensor, size=target_len, mode="linear", align_corners=True).squeeze(0).permute(1, 0).float()

            m_tensor = torch.from_numpy(mask).permute(1, 0).unsqueeze(0).float()
            res_m = torch.nn.functional.interpolate(m_tensor, size=target_len, mode="linear", align_corners=True).squeeze(0).permute(1, 0).float()
            res_mask = (res_m >= 0.5).to(torch.float32)

            features.append(res_t)
            masks.append(res_mask)
            labels.append(item["label"])
            sample_ids.append(item.get("sample_id", ""))

        return {
            "landmarks": torch.stack(features).float(),
            "mask": torch.stack(masks).float(),
            "label": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_ids,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--manifest", type=str, default="data/manifests/train_manifest.json")
    parser.add_argument("--validation-manifest", type=str, default="data/manifests/validation_manifest.json")
    parser.add_argument("--landmarks-dir", type=str, default="data/landmarks")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max samples")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, auto)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without training")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of data loader worker processes")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP) on CUDA")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit number of validation samples for faster epochs")
    parser.add_argument("--drive-backup", type=str, default=None, help="Auto-sync checkpoint copies to Drive directory")
    parser.add_argument("--run-name", type=str, default=None, help="Descriptive experiment run name to prevent overwriting")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast 1-epoch smoke test on small subset")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.smoke_test:
        logger.info("Executing fast smoke test (1 epoch, 8 samples, batch_size=4)...")
        config.training.epochs = 1
        args.limit = 8
        args.val_limit = 4
        config.training.batch_size = 4
    elif args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return

    set_seed(config.seed)
    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # Determine num_workers: default 2 on Linux (Colab), 0 on Windows to avoid IPC issues
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 2 if sys.platform != "win32" else 0

    pin_memory = (device.type == "cuda")
    logger.info("DataLoader config: batch_size=%d, num_workers=%d, pin_memory=%s, amp=%s",
                config.training.batch_size, num_workers, pin_memory, args.amp and (device.type == "cuda"))

    # 1. Load train samples
    train_samples = build_dataset_from_manifest(
        manifest_path=args.manifest,
        landmarks_dir=args.landmarks_dir,
        allowed_classes=config.dataset.labels,
    )
    if args.limit:
        train_samples = train_samples[: args.limit]

    if not train_samples:
        raise ValueError(f"No valid training samples found in {args.manifest} and {args.landmarks_dir}")

    # Determine vocabulary and label mapping
    if config.dataset.labels:
        class_names = list(config.dataset.labels)
    else:
        class_names = sorted({s.class_name for s in train_samples})

    label_to_index = {l: i for i, l in enumerate(class_names)}
    config.model.num_classes = len(class_names)

    train_dataset = SignLanguageDataset(train_samples, label_to_index=label_to_index)

    collator = PipelineCollator(sequence_length=config.dataset.sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.training.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # 2. Load validation samples
    val_loader = None
    if Path(args.validation_manifest).is_file():
        val_samples = build_dataset_from_manifest(
            manifest_path=args.validation_manifest,
            landmarks_dir=args.landmarks_dir,
            allowed_classes=class_names,
        )
        if args.val_limit:
            val_samples = val_samples[: args.val_limit]
        elif args.limit:
            val_samples = val_samples[: max(2, args.limit // 4)]
        if val_samples:
            val_dataset = SignLanguageDataset(val_samples, label_to_index=label_to_index)
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(config.training.batch_size, len(val_dataset)),
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )

    # 3. Build Model
    model = build_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    best_macro_f1 = -1.0

    if args.resume and Path(args.resume).is_file():
        logger.info("Resuming training from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
        else:
            model.load_state_dict(ckpt)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        run_name = "smoke_test"
    elif args.run_name:
        run_name = args.run_name
    else:
        run_name = config.model.name

    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
    last_ckpt_path = checkpoint_dir / f"{run_name}.pt"

    # Strictly protect the historical baseline checkpoint from accidental overwrite
    if best_ckpt_path.name == "robust_holistic_fusion_best.pt" and best_ckpt_path.is_file() and not getattr(args, "force_overwrite", False):
        import time
        safe_suffix = int(time.time())
        best_ckpt_path = checkpoint_dir / f"robust_holistic_fusion_run_{safe_suffix}_best.pt"
        last_ckpt_path = checkpoint_dir / f"robust_holistic_fusion_run_{safe_suffix}.pt"
        logger.info("Preserving baseline checkpoint: redirecting new outputs to %s", best_ckpt_path)

    # 4. Training Loop
    logger.info("Beginning training for %d epochs...", config.training.epochs)
    loss_history = []
    val_history = []
    results_dir = Path(config.results_dir)
    metrics_path = results_dir / "training_metrics.json"
    if args.resume and metrics_path.is_file():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                prev = json.load(f)
                loss_history = prev.get("loss_history", [])[:start_epoch]
                val_history = prev.get("validation_history", [])[:start_epoch]
                logger.info("Loaded previous history of %d epochs from %s", len(loss_history), metrics_path)
        except Exception as e:
            logger.warning("Could not read previous metrics history: %s", e)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) enabled on CUDA.")

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["landmarks"].to(device, dtype=torch.float32, non_blocking=pin_memory)
            masks = batch["mask"].to(device, dtype=torch.float32, non_blocking=pin_memory)
            targets = batch["label"].to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs, mask=masks)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()
                if config.training.gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.training.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / max(1, len(train_loader))
        loss_history.append(epoch_loss)

        # Validation step
        val_macro_f1 = 0.0
        val_metrics = {}
        should_validate = (val_loader is not None) and (
            ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == config.training.epochs)
        )
        if should_validate:
            model.eval()
            val_logits, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["landmarks"].to(device, dtype=torch.float32, non_blocking=pin_memory)
                    masks = batch["mask"].to(device, dtype=torch.float32, non_blocking=pin_memory)
                    targets = batch["label"].to(device, non_blocking=pin_memory)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(inputs, mask=masks)
                    val_logits.append(out.cpu())
                    val_targets.append(targets.cpu())
            if val_logits:
                val_metrics = evaluate_model(torch.cat(val_logits), torch.cat(val_targets))
                val_macro_f1 = float(val_metrics.get("macro_f1", 0.0))
                val_history.append(val_metrics)

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f",
            epoch + 1,
            config.training.epochs,
            epoch_loss,
            val_metrics.get("accuracy", 0.0),
            val_macro_f1,
        )

        # Build checkpoint metadata payload
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_macro_f1": max(best_macro_f1, val_macro_f1),
            "config": config.to_dict(),
            "model_name": config.model.name,
            "input_dim": config.model.input_dim,
            "num_classes": len(class_names),
            "class_names": class_names,
            "vocabulary_hash": compute_vocabulary_hash(class_names),
            "label_to_index": label_to_index,
            "sequence_length": config.dataset.sequence_length,
            "feature_representation": config.dataset.feature_representation,
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
        }

        # Save last checkpoint
        torch.save(checkpoint_payload, last_ckpt_path)

        # Save best checkpoint
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(checkpoint_payload, best_ckpt_path)
            logger.info("Saved new best model checkpoint to %s", best_ckpt_path)

        # Optional auto-sync to Google Drive directory
        if args.drive_backup:
            try:
                import shutil
                backup_dir = Path(args.drive_backup)
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_ckpt_path, backup_dir / last_ckpt_path.name)
                if val_macro_f1 == best_macro_f1 and best_ckpt_path.exists():
                    shutil.copy2(best_ckpt_path, backup_dir / best_ckpt_path.name)
            except Exception as e:
                logger.warning("Could not sync checkpoint to drive_backup: %s", e)

    # If no validation set was provided, also write best checkpoint
    if val_loader is None or best_macro_f1 < 0:
        torch.save(checkpoint_payload, best_ckpt_path)

    # 5. Save Experiment Metrics
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "loss_history": loss_history,
        "train_loss": loss_history,
        "validation_history": val_history,
        "val_macro_f1": [v.get("macro_f1", 0.0) for v in val_history],
        "val_accuracy": [v.get("accuracy", 0.0) for v in val_history],
        "best_macro_f1": best_macro_f1,
        "class_count": len(class_names),
        "classes": class_names,
    }
    metrics_path = results_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info("Training complete. Results saved to %s", metrics_path)

    if args.smoke_test:
        logger.info("Verifying checkpoint reload for smoke test...")
        reloaded_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        assert "model_state_dict" in reloaded_ckpt, "Reloaded checkpoint missing model_state_dict"
        test_model = build_model(config).to(device)
        test_model.load_state_dict(reloaded_ckpt["model_state_dict"])
        test_model.eval()
        dummy_in = torch.randn(1, config.dataset.sequence_length, config.model.input_dim).to(device)
        with torch.no_grad():
            dummy_out = test_model(dummy_in)
        assert dummy_out.shape[-1] == len(class_names), "Model output dimension mismatch"
        logger.info("SMOKE_TEST_VERIFIED: Forward pass, backward step, checkpoint save & reload fully operational.")


if __name__ == "__main__":
    main()
