# -*- coding: utf-8 -*-
"""
train_resnet.py - Improved 3D-ResNet(Model-2, 논문 Figure 3) 학습 스크립트

02_Model_Definition/models.py 의 ResNet3D(15-layer, 5 Residual Block x 3 unit)을
303명 PPMI T2 데이터로 학습합니다. train_ablation.py와 같은 데이터/전처리
(01_Preprocessing/전처리_ref21order_v1)와 같은 학습 안정화 기법(gradient clipping,
classifier 초기화 스케일 보정, val_loss 기준 best-checkpoint/조기종료)을 그대로 씁니다.

[논문 원문 인용 - Figure 3]
"This neural network architecture is composed of 15 layers, including an input
layer, two conv3D blocks, five residual blocks with max-pooling layers, a fully
connected layer, and a SoftMax layer. Each 'conv3D block' consists of a 3D conv
layer, BN, and ReLU activation layers. The residual block contains three residual
units, each comprised of two 3D Conv layers, BN, and ReLU activation layers...
skip connection... adds the input to the output of the last ReLU layer."

[2026-07-22 정정] 논문 Table 5("Comparison of training and validation details for 3D CNN
and 3D ResNet Models")에 ResNet 전용 학습 설정이 명시돼 있었음을 뒤늦게 발견 - 이전엔
CNN과 같은 lr(0.001)을 자체결정으로 그냥 가져다 썼는데, 표에는 3D ResNet의 Learning
rate가 **0.0001**(CNN의 0.001보다 10배 작음)로 별도 명시돼 있었음. lr=0.001로 첫 30epoch
전체(약 5.5시간) 실행에서 val_loss가 2600만까지 폭주하고 train_acc가 30epoch 내내 30%대에
머무는 심각한 불안정성을 겪었는데, 이 lr 오류가 원인 중 하나였을 가능성이 높음 - lr=0.0001로
정정 후 재실행.
- Table5는 이 외에도 k-fold cross validation=5(CNN/ResNet 둘 다)를 명시하는데, 이건 저희가
  지금까지 CNN 포함 전부 단일 holdout(70/15/15)으로 해온 것과 다른 방법론 - 별도로
  k-fold 전환 여부를 판단해야 함(이 스크립트에는 아직 미반영).
- batch_size=32, weight_decay=0.0001(Table5 "L2 Regularization"과 일치)은 유지.
  batch_size 자체는 Table5에 명시 없어 자체결정 그대로.

[논문 목표치] Model-2(ResNet, FC-4 특징을 Gradient Boosting으로 분류) 90.0%
(주의: 이건 GB 분류기를 쓴 결과이고, 이 스크립트는 ResNet 자체의 SoftMax 분류
정확도를 봄 - 직접 비교 불가, 참고용)
"""
import argparse
import os
import sys
import json
import time

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from final_resnet import ResNet3D
from train_ablation import (
    build_loaders, build_loaders_from_samples, calibrate_classifier_scale, run_epoch,
    evaluate_final, train_loop, _macro_prf,
)
from dataset import get_kfold_splits_with_val

PAPER_TARGET_ACC = 0.90  # Model-2, GB 분류기 적용시 (참고용 - 직접비교 불가)


def run_kfold_resnet(args):
    """[2026-07-23 추가] train_ablation.run_kfold()와 동일한 목적(Table5 "k-fold CV=5"
    재현) - ResNet은 VARIANT_MODELS 딕셔너리 구조를 안 쓰므로 별도 함수로 둠. 학습 로직
    (train_loop) 자체는 CNN ablation과 완전히 동일한 함수를 재사용."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold_summaries = []
    fold_records = []
    kfold_t0 = time.time()

    for fold_idx, (train_samples, val_samples, test_samples) in enumerate(
        get_kfold_splits_with_val(args.csv_path, k=args.kfold, seed=args.seed)
    ):
        fold_no = fold_idx + 1
        print(f"\n{'='*20} [Fold {fold_no}/{args.kfold}] resnet3d {'='*20}")
        torch.manual_seed(args.seed + fold_idx)

        train_loader, val_loader, test_loader = build_loaders_from_samples(
            train_samples, val_samples, test_samples, args.image_dir, args.batch_size
        )

        model = ResNet3D(num_classes=3).to(device)
        n_params = sum(p_.numel() for p_ in model.parameters())

        loop_result = train_loop(
            model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, grad_clip_norm=args.grad_clip_norm, patience=args.patience,
            min_epochs=args.min_epochs, calibrate_init=args.calibrate_init,
            log_prefix=f"  [Fold {fold_no}] ",
        )

        criterion = nn.CrossEntropyLoss()
        test_result = evaluate_final(model, test_loader, device, criterion, f"Fold {fold_no} Test")
        test_result["best_epoch"] = loop_result["best_epoch"]
        test_result["best_val_acc"] = loop_result["best_val_acc"]

        fold_summary = {
            "fold": fold_no, "n_params": n_params,
            "actual_epochs": loop_result["actual_epochs"], "best_epoch": loop_result["best_epoch"],
            "best_val_acc": round(loop_result["best_val_acc"], 4),
            "test_accuracy": round(test_result["accuracy"], 4),
            "test_precision_macro": round(test_result.get("precision_macro", float("nan")), 4),
            "test_recall_macro": round(test_result.get("recall_macro", float("nan")), 4),
            "test_f1_macro": round(test_result.get("f1_macro", float("nan")), 4),
            "elapsed_sec": round(loop_result["elapsed"], 1),
        }
        fold_summaries.append(fold_summary)
        fold_records.append({
            "fold": fold_no, "summary": fold_summary,
            "history": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                for row in loop_result["history"]
            ],
        })
        print(f"  [Fold {fold_no} 결과] Acc={fold_summary['test_accuracy']*100:.2f}% "
              f"Precision={fold_summary['test_precision_macro']*100:.2f}% "
              f"Recall={fold_summary['test_recall_macro']*100:.2f}% "
              f"F1={fold_summary['test_f1_macro']*100:.2f}%  best_epoch={fold_summary['best_epoch']}")

        checkpoints_dir = os.path.join(_ROOT, "03_Model_Training", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        ts_fs = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(
            checkpoints_dir,
            f"resnet3d_kfold{args.kfold}_fold{fold_no}_{ts_fs}_acc{fold_summary['test_accuracy']*100:.1f}.pt",
        )
        torch.save({
            "variant": "resnet3d", "model_state_dict": loop_result["best_state"],
            "num_classes": 3, "class_names": ["Control", "Prodromal", "PD"],
            "kfold": args.kfold, "fold": fold_no,
            "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                             "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
                             "calibrate_init": args.calibrate_init, "seed": args.seed},
            "test_accuracy": test_result["accuracy"],
            "test_precision_macro": test_result.get("precision_macro"),
            "test_recall_macro": test_result.get("recall_macro"),
            "test_f1_macro": test_result.get("f1_macro"),
            "best_epoch": loop_result["best_epoch"],
        }, ckpt_path)

    kfold_elapsed = time.time() - kfold_t0

    def _mean_std(key):
        vals = [fs[key] for fs in fold_summaries if fs[key] == fs[key]]
        if not vals:
            return float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, var ** 0.5

    agg = {}
    for key in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]:
        mean, std = _mean_std(key)
        agg[key + "_mean"] = round(mean, 4)
        agg[key + "_std"] = round(std, 4)

    print(f"\n{'='*20} [K-Fold(k={args.kfold}) 집계 결과 - resnet3d] {'='*20}")
    print(f"Accuracy : {agg['test_accuracy_mean']*100:.2f}% ± {agg['test_accuracy_std']*100:.2f}%p")
    print(f"Precision: {agg['test_precision_macro_mean']*100:.2f}% ± {agg['test_precision_macro_std']*100:.2f}%p")
    print(f"Recall   : {agg['test_recall_macro_mean']*100:.2f}% ± {agg['test_recall_macro_std']*100:.2f}%p")
    print(f"F1       : {agg['test_f1_macro_mean']*100:.2f}% ± {agg['test_f1_macro_std']*100:.2f}%p")
    print(f"총 소요시간: {kfold_elapsed:.1f}s ({kfold_elapsed/3600:.2f}시간)")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fs = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{agg['test_accuracy_mean']*100:.1f}"
    out_path = os.path.join(results_dir, f"kfold_resnet3d_k{args.kfold}_{timestamp_fs}_acc{acc_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "timestamp": timestamp, "variant": "resnet3d", "kfold": args.kfold,
                "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                                 "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
                                 "calibrate_init": args.calibrate_init, "seed": args.seed,
                                 "patience": args.patience, "min_epochs": args.min_epochs},
                "n_folds": args.kfold, "elapsed_sec": round(kfold_elapsed, 1),
                **agg,
            },
            "fold_summaries": fold_summaries,
            "fold_records": fold_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"K-Fold 집계 결과 JSON 저장: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Improved 3D-ResNet(Model-2) 학습")
    p.add_argument("--csv_path", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv"))
    p.add_argument("--image_dir", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--no_calibrate_init", action="store_false", dest="calibrate_init", default=True)
    # [2026-07-26 추가] 과적합 대응(Best Validation 시트에서 Train Acc 65.09% vs
    # Val Acc 46.67% 확인) - Dropout + Data Augmentation 병행 시도. 둘 다 논문
    # Table5에 없는 자체결정 항목이라 기본값 없이 명시적으로 켜야 적용됨.
    p.add_argument("--dropout", type=float, default=0.0,
                   help="ResNet3D의 FC-4->classifier 사이 Dropout 확률(기본 0=미적용). 과적합 대응시 0.5 권장")
    p.add_argument("--augment", action="store_true", default=False,
                   help="train에 dataset.augment_volume_3d(회전/뒤집기/스케일/노이즈) 적용")
    # [2026-07-27 추가] train_ablation.py에는 이미 있던 옵션인데 이 스크립트엔 빠져
    # 있었음 - 30epoch 내내 val_acc가 26~53%로 크게 요동치는데 val_loss 최저 기준으로
    # 고르면 중간 정도 성능(46.67%)인 epoch이 뽑히고 더 나은 epoch(51.11%)을 지나치는
    # 문제를 실측으로 확인(dropout+augment 재학습 결과) - Variant3에서 이미 겪었던
    # 것과 같은 패턴.
    p.add_argument("--checkpoint_metric", type=str, default="val_loss", choices=["val_loss", "val_acc"],
                   help="best-checkpoint 선택 기준. val_loss(기본) 또는 val_acc")
    # [2026-07-23 추가] train_ablation.py와 동일한 목적 - Table5 "k-fold CV=5" 반영
    p.add_argument("--kfold", type=int, default=None, choices=[5, 10, 15],
                   help="지정 시 Table5의 k-fold CV를 재현(fold마다 학습, 결과 집계). 미지정시 기존 단일 holdout")
    args = p.parse_args()

    if args.kfold:
        run_kfold_resnet(args)
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = build_loaders(
        args.csv_path, args.image_dir, args.batch_size, args.seed,
        use_augmentation=args.augment,
    )

    model = ResNet3D(num_classes=3, dropout=args.dropout).to(device)
    if args.dropout > 0:
        print(f"[Dropout] FC-4->classifier 사이 p={args.dropout}")
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[모델] resnet3d (ResNet3D), 파라미터 수={n_params:,}")

    # [2026-07-23 리팩터] 학습 루프를 train_ablation.train_loop()로 통일 - 이제 epoch별
    # precision/recall/f1(macro)도 함께 기록됨(친구 쪽 training_history_ResNet.csv와
    # 같은 형식으로 비교 가능해짐).
    loop_result = train_loop(
        model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, grad_clip_norm=args.grad_clip_norm, patience=args.patience,
        min_epochs=args.min_epochs, calibrate_init=args.calibrate_init,
        checkpoint_metric=args.checkpoint_metric,
    )
    history = loop_result["history"]
    best_epoch = loop_result["best_epoch"]
    best_val_acc = loop_result["best_val_acc"]
    best_val_loss = loop_result["best_val_loss"]
    best_state = loop_result["best_state"]
    elapsed = loop_result["elapsed"]
    actual_epochs = loop_result["actual_epochs"]
    print(f"\n총 학습 시간: {elapsed:.1f}s")
    print(f"\n[체크포인트 선택] Best validation epoch = {best_epoch}/{actual_epochs} "
          f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, {best_val_acc*100:.2f}%) - 이 시점 가중치로 test 평가")

    criterion = nn.CrossEntropyLoss()
    test_result = evaluate_final(model, test_loader, device, criterion, "Test")
    test_result["best_epoch"] = best_epoch
    test_result["best_val_acc"] = best_val_acc

    print("\n" + "=" * 60)
    print(f"참고: 논문 Model-2(ResNet FC-4 특징 -> Gradient Boosting 분류) 목표 정확도 = {PAPER_TARGET_ACC*100:.1f}%")
    print("(이 스크립트는 ResNet 자체 SoftMax 분류 결과라 GB 결과와 직접 비교 불가 - 참고용)")
    print(f"우리 ResNet SoftMax 분류 Accuracy = {test_result['accuracy']*100:.2f}%")
    print("=" * 60)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fs = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{test_result['accuracy']*100:.1f}"
    out_path = os.path.join(results_dir, f"resnet3d_{timestamp_fs}_acc{acc_tag}.json")

    summary = {
        "timestamp": timestamp,
        "model": "resnet3d",
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
            "calibrate_init": args.calibrate_init, "seed": args.seed,
            "patience": args.patience, "min_epochs": args.min_epochs,
            "dropout": args.dropout, "augment": args.augment,
            "checkpoint_metric": args.checkpoint_metric,
        },
        "n_params": n_params,
        "actual_epochs": actual_epochs,
        "early_stopped": actual_epochs < args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_precision_macro": round(test_result.get("precision_macro", float("nan")), 4),
        "test_recall_macro": round(test_result.get("recall_macro", float("nan")), 4),
        "test_f1_macro": round(test_result.get("f1_macro", float("nan")), 4),
        "elapsed_sec": round(elapsed, 1),
        "paper_model2_gb_accuracy": PAPER_TARGET_ACC,
    }
    history_rounded = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in epoch_row.items()}
        for epoch_row in history
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "test_result": test_result, "history": history_rounded}, f,
                   ensure_ascii=False, indent=2)
    print(f"상세 결과 JSON 저장: {out_path}")

    checkpoints_dir = os.path.join(_ROOT, "03_Model_Training", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"resnet3d_{timestamp_fs}_acc{acc_tag}.pt")
    torch.save({
        "variant": "resnet3d",
        "model_state_dict": best_state,
        "num_classes": 3,
        "class_names": ["Control", "Prodromal", "PD"],
        "hyperparams": summary["hyperparams"],
        "test_accuracy": test_result["accuracy"],
        "test_precision_macro": test_result.get("precision_macro"),
        "test_recall_macro": test_result.get("recall_macro"),
        "test_f1_macro": test_result.get("f1_macro"),
        "best_epoch": best_epoch,
        "timestamp": timestamp,
    }, ckpt_path)
    print(f"체크포인트 저장 (특징추출/추론용): {ckpt_path}")


if __name__ == "__main__":
    main()
