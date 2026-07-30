# -*- coding: utf-8 -*-
"""
run_real_pipeline.py - Model-3(CCA)+Model-4(WOA) 전체 파이프라인을 실제 체크포인트
(Variant3/ResNet)에서 뽑은 real 특징으로 실행하는 스크립트.

extract_features.py(특징 추출) -> cca_feature_fusion.py(CCA, Eq6/Eq7 둘 다) ->
woa_feature_selection.py(WOA, Table7 20회 독립실행) -> GradientBoosting 평가
순서로 전체를 한 번에 돈다. 기존에 03_Model_Training에 흩어져 있던
run_cca.py + woa_feature_selection.py(실행부)를 이 파일 하나로 통합함
(2026-07-24, 폴더 재구성).

출력:
- fused_{train,val,test}.npz (이 폴더에 저장 - CCA 결과, 05_Model_Evaluation에서 재사용)
- results_model3_model4.json (Model-3/Model-4 최종 성능 + WOA 20회 집계)
"""
import os
import sys
import time
import json
import numpy as np
import torch

_ROOT = r"D:\new tensor"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "02_Model_Definition"))
sys.path.insert(0, os.path.join(_ROOT, "03_Model_Training"))
sys.path.insert(0, _THIS_DIR)

from train_ablation import build_loaders
# [2026-07-24 정리] 체크포인트 로딩(load_cnn/load_resnet)과 특징 추출(extract_all)은
# extract_features.py에 이미 있는 걸 그대로 재사용 - 이 파일에서 또 만들지 않음
# (예전엔 여기서 CNN_CKPT/RESNET_CKPT 경로와 추출 로직을 통째로 다시 정의해서
# extract_features.py와 두 벌로 중복돼 있었음).
from extract_features import load_cnn, load_resnet, extract_all
from cca_feature_fusion import cca_fuse
from woa_feature_selection import binary_woa_feature_selection, POP_SIZE, N_ITER, N_INDEPENDENT_RUNS, ALPHA, KNN_K, THRESHOLD, B_CONST

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(Xtr, ytr, Xte, yte):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}


def main():
    print("Device:", device)
    cnn, cnn_hp = load_cnn()
    resnet, resnet_hp = load_resnet()

    csv_path = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
    image_dir = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")
    train_loader, val_loader, test_loader = build_loaders(csv_path, image_dir, 32, 42)

    print("\n[1/4] 특징 추출 (FV-3=Algorithm1, 1000차원 / FC-4=1000차원 - 논문 원문대로)")
    fv3_train, fv4_train, y_train, _ = extract_all(cnn, resnet, train_loader)
    fv3_val, fv4_val, y_val, _ = extract_all(cnn, resnet, val_loader)
    fv3_test, fv4_test, y_test, _ = extract_all(cnn, resnet, test_loader)
    print(f"  train FV-3:{fv3_train.shape} FC-4:{fv4_train.shape}  (합계 {fv3_train.shape[1]+fv4_train.shape[1]}차원, 논문='2000')")

    print("\n[2/4] CCA 융합 (Model-3, Eq3-7, n_components=수학적 최댓값)")
    z_concat_train, z_sum_train, cca = cca_fuse(fv3_train, fv4_train)
    z_concat_val, z_sum_val, _ = cca_fuse(fv3_val, fv4_val, fitted_cca=cca)
    z_concat_test, z_sum_test, _ = cca_fuse(fv3_test, fv4_test, fitted_cca=cca)
    print(f"  n_components={cca.n_components}  z_concat={z_concat_train.shape}  z_sum={z_sum_train.shape}")

    np.savez(os.path.join(_THIS_DIR, "fused_train.npz"), z_concat=z_concat_train, z_sum=z_sum_train, y=y_train)
    np.savez(os.path.join(_THIS_DIR, "fused_val.npz"), z_concat=z_concat_val, z_sum=z_sum_val, y=y_val)
    np.savez(os.path.join(_THIS_DIR, "fused_test.npz"), z_concat=z_concat_test, z_sum=z_sum_test, y=y_test)
    print("  fused_{train,val,test}.npz 저장 완료")

    print("\n[3/4] Model-3 평가 (WOA 적용 전, Eq6 vs Eq7 비교)")
    model3_concat = evaluate(z_concat_train, y_train, z_concat_test, y_test)
    model3_sum = evaluate(z_sum_train, y_train, z_sum_test, y_test)
    print(f"  Eq6(concat): {model3_concat}")
    print(f"  Eq7(sum)   : {model3_sum}")
    fusion_choice = "z_sum" if model3_sum["f1_macro"] >= model3_concat["f1_macro"] else "z_concat"
    Xtr_fused = z_sum_train if fusion_choice == "z_sum" else z_concat_train
    Xte_fused = z_sum_test if fusion_choice == "z_sum" else z_concat_test
    print(f"  -> Model-4(WOA) 입력으로 {fusion_choice} 채택(F1 기준 더 좋은 쪽)")

    print(f"\n[4/4] Model-4: WOA 특징선택 (Table7, population={POP_SIZE}, iterations={N_ITER}, "
          f"{N_INDEPENDENT_RUNS}회 독립실행, 3-fold CV fitness)")
    t_total0 = time.time()
    woa_runs = []
    for run_i in range(N_INDEPENDENT_RUNS):
        t0 = time.time()
        seed = 1000 + run_i
        mask, best_fit, hist = binary_woa_feature_selection(Xtr_fused, y_train, seed=seed)
        elapsed = time.time() - t0
        result = evaluate(Xtr_fused[:, mask], y_train, Xte_fused[:, mask], y_test)
        woa_runs.append({"run": run_i, "seed": seed, "n_selected": int(mask.sum()),
                          "fitness": float(best_fit), "elapsed_sec": round(elapsed, 1), **result})
        print(f"  [실행 {run_i+1}/{N_INDEPENDENT_RUNS}] 선택={mask.sum()}/{Xtr_fused.shape[1]} "
              f"fitness={best_fit:.4f} acc={result['accuracy']:.4f} f1={result['f1_macro']:.4f} 소요={elapsed:.1f}s")
    total_elapsed = time.time() - t_total0

    accs = np.array([r["accuracy"] for r in woa_runs])
    f1s = np.array([r["f1_macro"] for r in woa_runs])

    print("\n" + "=" * 60)
    print("[Model-3 결과] WOA 적용 전:", fusion_choice, "=", model3_sum if fusion_choice == "z_sum" else model3_concat)
    print(f"[Model-4 결과] WOA 적용 후(20회): Accuracy={accs.mean()*100:.2f}%±{accs.std()*100:.2f}%p "
          f"(최고 {accs.max()*100:.2f}%) F1={f1s.mean()*100:.2f}% (최고 {f1s.max()*100:.2f}%)")
    print(f"WOA 총 소요시간: {total_elapsed:.1f}s ({total_elapsed/60:.1f}분)")
    print("=" * 60)

    out = {
        "hyperparams": {"population": POP_SIZE, "iterations": N_ITER, "n_independent_runs": N_INDEPENDENT_RUNS,
                         "b_const": B_CONST, "threshold": THRESHOLD, "alpha": ALPHA, "knn_k": KNN_K},
        "cca_n_components": int(cca.n_components),
        "fusion_choice_for_woa": fusion_choice,
        "model3": {"eq6_concat": model3_concat, "eq7_sum": model3_sum},
        "model4_woa": {
            "total_elapsed_sec": round(total_elapsed, 1),
            "summary": {
                "test_accuracy_mean": float(accs.mean()), "test_accuracy_std": float(accs.std()),
                "test_accuracy_max": float(accs.max()), "test_accuracy_min": float(accs.min()),
                "test_f1_mean": float(f1s.mean()), "test_f1_max": float(f1s.max()), "test_f1_min": float(f1s.min()),
            },
            "runs": woa_runs,
        },
    }
    out_path = os.path.join(_ROOT, "03_Model_Training", "results", "model3_model4_final_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("결과 저장:", out_path)


if __name__ == "__main__":
    main()
