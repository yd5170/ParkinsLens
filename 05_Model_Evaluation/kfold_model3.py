# -*- coding: utf-8 -*-
"""
Model-3(CCA 융합, WOA 전) k-fold(5/10/15) 평가 - 논문 Table6 방식 재현.

CNN(Variant3)/ResNet은 이미 학습 완료된 고정 체크포인트를 특징 추출기로만 재사용
(재학습 없음 - dataset.py get_kfold_splits로 만든 fold별 train/test 인원에 대해
매번 새로 forward pass만 수행). CCA는 각 fold의 train 데이터로만 fit(fold별로
매번 새로 학습 - test 오염 방지, 진짜 k-fold 방식).
"""
import os
import sys
import time
import json
import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ["Control", "Prodromal", "PD"]
SINGLE_HOLDOUT_BASELINE = {  # 기존 단일 train/val/test(Eq7 합산) 결과 - 비교용
    "accuracy": 0.4565, "precision_macro": 0.4744, "recall_macro": 0.4518, "f1_macro": 0.4467,
}

_ROOT = r"D:\new tensor"
sys.path.insert(0, os.path.join(_ROOT, "02_Model_Definition"))
sys.path.insert(0, os.path.join(_ROOT, "03_Model_Training"))
# [2026-07-24 이동] extract_features.py가 04_Feature_Engineering으로 옮겨져 경로 추가
sys.path.insert(0, os.path.join(_ROOT, "04_Feature_Engineering"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_kfold_splits, PPMIT2Dataset
from torch.utils.data import DataLoader
# [2026-07-24 정리] 체크포인트 로딩과 특징 추출은 extract_features.py 걸 그대로
# 재사용(load_cnn/load_resnet, Algorithm1 적용은 extract_all 내부에서 이미 처리) -
# 이 파일에서 모델 로딩·forward 재구현을 중복하지 않음. 다만
# extract_features.extract_all()은 DataLoader 하나를 받는데 여기는 fold마다 새
# DataLoader가 필요해서 extract_features_for_samples()만 이 파일에 남겨둠
# (loader 준비 + extract_all 호출).
from extract_features import load_cnn, load_resnet, extract_all

CSV_PATH = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
IMAGE_DIR = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    cnn, _ = load_cnn()
    resnet, _ = load_resnet()
    return cnn, resnet


def extract_features_for_samples(cnn, resnet, samples):
    exists = lambda s: os.path.exists(os.path.join(IMAGE_DIR, f"{s['sample_id']}.nii.gz"))
    samples = [s for s in samples if exists(s)]
    loader = DataLoader(PPMIT2Dataset(samples, IMAGE_DIR), batch_size=32, shuffle=False, num_workers=0)
    fv3_all, fv4_all, y_all, _ = extract_all(cnn, resnet, loader)
    return fv3_all, fv4_all, y_all


def run_kfold(cnn, resnet, k):
    fold_results = []
    t0 = time.time()
    for fold_idx, (train_samples, test_samples) in enumerate(get_kfold_splits(CSV_PATH, k=k, seed=42)):
        fv3_tr, fv4_tr, y_tr = extract_features_for_samples(cnn, resnet, train_samples)
        fv3_te, fv4_te, y_te = extract_features_for_samples(cnn, resnet, test_samples)

        # [수정] n_components를 fold 크기 최댓값까지 그대로 쓰면 sklearn CCA의
        # power-iteration SVD가 간헐적으로 수렴 실패함(특정 fold의 데이터 조건에 따라
        # 수치적으로 불안정해짐 - fold마다 재현되는 게 아니라 일부 fold에서만 발생).
        # 211부터 시작해 실패하면 점점 줄여가며 재시도(자체결정 - 실패 fold를 그냥
        # 건너뛰지 않고 가능한 한 큰 n_components로 성공시키는 걸 우선).
        n_components_try = min(fv3_tr.shape[1], fv4_tr.shape[1], len(y_tr) - 1, 211)
        wa_tr = wb_tr = wa_te = wb_te = None
        n_components = n_components_try
        while n_components >= 10:
            try:
                cca = CCA(n_components=n_components, max_iter=2000)
                cca.fit(fv3_tr, fv4_tr)
                wa_tr, wb_tr = cca.transform(fv3_tr, fv4_tr)
                wa_te, wb_te = cca.transform(fv3_te, fv4_te)
                break
            except np.linalg.LinAlgError:
                print(f"    [경고] n_components={n_components}에서 SVD 미수렴 - "
                      f"{n_components - 20}로 줄여 재시도")
                n_components -= 20
        if wa_tr is None:
            print(f"    [실패] k={k} fold {fold_idx+1} - CCA 수렴 불가로 이 fold 건너뜀(요약에서 제외)")
            continue
        z_tr = wa_tr + wb_tr   # Eq(7) 합산 - Eq6/7 비교에서 검증된 더 나은 방식
        z_te = wa_te + wb_te

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(z_tr, y_tr)
        pred = clf.predict(z_te)
        acc = accuracy_score(y_te, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
        # 클래스별(macro 아닌) precision/recall/f1 - Prodromal이 유독 약했던 패턴이
        # k-fold에서도 반복되는지 확인하기 위해 추가
        p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
            y_te, pred, average=None, labels=[0, 1, 2], zero_division=0
        )

        fold_results.append({
            "fold": fold_idx, "n_train": len(y_tr), "n_test": len(y_te),
            "n_components": n_components,
            "accuracy": float(acc), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": float(p_cls[i]), "recall": float(r_cls[i]),
                    "f1": float(f1_cls[i]), "support": int(support_cls[i]),
                } for i in range(3)
            },
            "y_true": y_te.tolist(), "y_pred": pred.tolist(),  # 전체 fold 합산 confusion matrix용
        })
        print(f"  [k={k}, fold {fold_idx+1}/{k}] n_train={len(y_tr)} n_test={len(y_te)} "
              f"acc={acc:.4f} f1={f1:.4f}")

    elapsed = time.time() - t0
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    ps = [r["precision_macro"] for r in fold_results]
    rs = [r["recall_macro"] for r in fold_results]
    n_skipped = k - len(fold_results)

    # 전체 fold의 예측을 다 합쳐서 confusion matrix 계산(fold별로 따로 보는 것보다
    # 종합된 오차 패턴이 더 명확함)
    all_y_true = [v for r in fold_results for v in r["y_true"]]
    all_y_pred = [v for r in fold_results for v in r["y_pred"]]
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2]).tolist()

    # 클래스별 지표도 전체 fold 평균으로 집계
    per_class_agg = {}
    for cls in CLASS_NAMES:
        per_class_agg[cls] = {
            "precision_mean": float(np.mean([r["per_class"][cls]["precision"] for r in fold_results])),
            "recall_mean": float(np.mean([r["per_class"][cls]["recall"] for r in fold_results])),
            "f1_mean": float(np.mean([r["per_class"][cls]["f1"] for r in fold_results])),
        }

    summary = {
        "k": k, "elapsed_sec": round(elapsed, 1),
        "n_folds_completed": len(fold_results), "n_folds_skipped": n_skipped,
        "accuracy_mean": float(np.mean(accs)), "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)), "f1_mean": float(np.mean(f1s)),
        "accuracy_std": float(np.std(accs)),
        "confusion_matrix": cm,  # 3x3, 행=실제 클래스, 열=예측 클래스 (Control/Prodromal/PD 순)
        "per_class_mean": per_class_agg,
        "vs_single_holdout_baseline": {
            k2: round(summary_val - SINGLE_HOLDOUT_BASELINE[k2], 4)
            for k2, summary_val in [
                ("accuracy", float(np.mean(accs))), ("precision_macro", float(np.mean(ps))),
                ("recall_macro", float(np.mean(rs))), ("f1_macro", float(np.mean(f1s))),
            ]
        },
        "folds": fold_results,
    }
    skip_note = f" ({n_skipped}개 fold는 CCA 미수렴으로 제외)" if n_skipped else ""
    print(f"[k={k} 요약] Accuracy={summary['accuracy_mean']:.4f}(+-{summary['accuracy_std']:.4f}) "
          f"Precision={summary['precision_mean']:.4f} Recall={summary['recall_mean']:.4f} "
          f"F1={summary['f1_mean']:.4f}  (소요 {elapsed:.1f}s){skip_note}")
    print(f"  클래스별 F1: Control={per_class_agg['Control']['f1_mean']:.4f} "
          f"Prodromal={per_class_agg['Prodromal']['f1_mean']:.4f} PD={per_class_agg['PD']['f1_mean']:.4f}")
    print(f"  단일 holdout 대비 차이: Accuracy {summary['vs_single_holdout_baseline']['accuracy']:+.4f}")
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cnn, resnet = load_models()
    print("모델 로드 완료. Model-3(CCA+GB, WOA 전) k-fold 평가 시작.\n")

    all_summaries = {}
    for k in (5, 10, 15):
        print(f"=== k={k} ===")
        all_summaries[k] = run_kfold(cnn, resnet, k)
        print()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model3_kfold_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")

    print("\n=== 논문 Table6(Model-3) 대비 ===")
    paper = {5: (0.916, 0.886, 0.935, 0.905), 10: (0.924, 0.882, 0.928, 0.918), 15: (0.948, 0.904, 0.956, 0.938)}
    for k in (5, 10, 15):
        s = all_summaries[k]
        pa, pr, pp, pf = paper[k]
        print(f"k={k}: 우리 Acc={s['accuracy_mean']:.3f}(논문 {pa}) "
              f"Recall={s['recall_mean']:.3f}(논문 {pr}) Precision={s['precision_mean']:.3f}(논문 {pp}) "
              f"F1={s['f1_mean']:.3f}(논문 {pf})")

    print("\n=== 클래스별 F1 (전체 fold 평균) ===")
    for k in (5, 10, 15):
        pc = all_summaries[k]["per_class_mean"]
        print(f"k={k}: Control={pc['Control']['f1_mean']:.3f} "
              f"Prodromal={pc['Prodromal']['f1_mean']:.3f} PD={pc['PD']['f1_mean']:.3f}")

    print("\n=== 합산 Confusion Matrix (행=실제, 열=예측 - Control/Prodromal/PD 순) ===")
    for k in (5, 10, 15):
        print(f"k={k}:")
        for row_name, row in zip(CLASS_NAMES, all_summaries[k]["confusion_matrix"]):
            print(f"  실제={row_name:9s} 예측[Control={row[0]:3d} Prodromal={row[1]:3d} PD={row[2]:3d}]")

    print("\n=== 단일 holdout(45.65%) 대비 k-fold 평균 차이 ===")
    for k in (5, 10, 15):
        v = all_summaries[k]["vs_single_holdout_baseline"]
        print(f"k={k}: Accuracy {v['accuracy']:+.4f}, F1 {v['f1_macro']:+.4f}")
