# -*- coding: utf-8 -*-
"""
train_cnn3d.py - cnn3d_figure2.py의 CNN3D_Figure2(Model-1, 1차 모델·Figure2 문자
그대로 재현) 학습 스크립트.

[2026-07-26 파일 분리] 원래 models.py 안에 최종 모델(Variant3)과 같이 있던 1차 모델을
cnn3d_figure2.py로 분리하고, 최종 모델은 final_cnn.py(CNN3D=CNN3D_Variant3 별칭)와
final_resnet.py(ResNet3D)로 옮김 - 파일명만으로 어느 게 최종본인지 알 수 있게 하기
위함. 상세: 02_Model_Definition/README.md.

train_resnet.py와 동일한 구조(공용 train_ablation.build_loaders/train_loop/
evaluate_final 재사용)로 작성. CNN3D는 ablation_models.py의 Base/Variant1~3와
달리 Table3 Study2 그리드서치 대상이 아니라서 논문이 명시한 batch_size가 없음 -
learning rate는 Table5의 "3D CNN Learning rate=0.001"을 그대로 사용(ResNet만
별도로 0.0001), batch_size=32는 자체 선택.

[2026-07-23 참고 - profile_cnn3d_layers.py 실측 결과] batch=32 기준 GPU(RTX 4070,
12GB) 메모리 사용량이 11.7GB까지 올라가고, backward+step이 forward보다 8배 이상
느림(정상은 1.5~3배) - 메모리 압박으로 추정됨. 1epoch 828초(13.8분), 30epoch 약
6.9시간 예상. 사용자가 batch를 줄여 먼저 확인하는 방안 대신 batch=32 그대로 바로
실행하기로 결정함(2026-07-23) - 실제 학습에서도 이 속도 이슈가 재현되는지는 이
실행으로 확인됨.
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

from cnn3d_figure2 import CNN3D_Figure2 as CNN3D
from train_ablation import build_loaders, evaluate_final, train_loop


def main():
    p = argparse.ArgumentParser(description="CNN3D_Figure2Legacy(Model-1, 1차 모델) 학습")
    p.add_argument("--csv_path", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv"))
    p.add_argument("--image_dir", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)  # 논문 미명시(Table3 Study2 대상 아님) - 자체 선택
    p.add_argument("--lr", type=float, default=0.001)      # Table5 "3D CNN Learning rate"
    p.add_argument("--weight_decay", type=float, default=0.0001)  # Table5 L2
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--no_calibrate_init", action="store_false", dest="calibrate_init", default=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = build_loaders(args.csv_path, args.image_dir, args.batch_size, args.seed)

    model = CNN3D(num_classes=3).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[모델] cnn3d (CNN3D_Figure2Legacy, Model-1 1차 모델), 파라미터 수={n_params:,}")

    loop_result = train_loop(
        model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, grad_clip_norm=args.grad_clip_norm, patience=args.patience,
        min_epochs=args.min_epochs, calibrate_init=args.calibrate_init,
    )
    history = loop_result["history"]
    best_epoch = loop_result["best_epoch"]
    best_val_acc = loop_result["best_val_acc"]
    best_val_loss = loop_result["best_val_loss"]
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
    print(f"CNN3D_Figure2Legacy(Model-1, 1차 모델) SoftMax 분류 Accuracy = {test_result['accuracy']*100:.2f}%")
    print("(참고: 이 모델은 Table3 Study2 그리드서치 대상이 아니라 논문 비교 목표치 없음)")
    print("=" * 60)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fs = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{test_result['accuracy']*100:.1f}"
    out_path = os.path.join(results_dir, f"cnn3d_{timestamp_fs}_acc{acc_tag}.json")

    summary = {
        "timestamp": timestamp,
        "model": "cnn3d",
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
            "calibrate_init": args.calibrate_init, "seed": args.seed,
            "patience": args.patience, "min_epochs": args.min_epochs,
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
        "note": "Table3 Study2 그리드서치 대상 아님(ablation variant가 아닌 1차/Figure2 모델) - 논문 비교 목표치 없음",
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
    ckpt_path = os.path.join(checkpoints_dir, f"cnn3d_{timestamp_fs}_acc{acc_tag}.pt")
    torch.save({
        "variant": "cnn3d",
        "model_state_dict": loop_result["best_state"],
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
    print(f"체크포인트 저장: {ckpt_path}")


if __name__ == "__main__":
    main()
