import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# [2026-07-24 이동] 04_Feature_Engineering/run_real_pipeline.py가 fused_*.npz를
# 저장하는 위치로 맞춤(이전엔 이 파일 자신의 폴더를 봤음 - 폴더 재구성으로 변경).
_ROOT = r"D:\new tensor"
_FUSED_DIR = os.path.join(_ROOT, "04_Feature_Engineering")

train = np.load(os.path.join(_FUSED_DIR, "fused_train.npz"))
val = np.load(os.path.join(_FUSED_DIR, "fused_val.npz"))
test = np.load(os.path.join(_FUSED_DIR, "fused_test.npz"))

CLASS_NAMES = ["Control", "Prodromal", "PD"]


def run(key, label):
    Xtr, ytr = train[key], train["y"]
    Xval, yval = val[key], val["y"]
    Xte, yte = test[key], test["y"]

    clf = GradientBoostingClassifier(random_state=42)  # 논문 미기재 - sklearn 기본값(DEVIATIONS.md 5번 항목과 동일 근거)
    clf.fit(Xtr, ytr)

    print(f"\n{'='*60}\n[{label}] (dim={Xtr.shape[1]})\n{'='*60}")
    for split_name, X, y in [("Val", Xval, yval), ("Test", Xte, yte)]:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        print(f"{split_name}: Accuracy={acc:.4f} Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
        if split_name == "Test":
            print(classification_report(y, pred, target_names=CLASS_NAMES, zero_division=0))
    return clf


run("z_concat", "Eq(6) concat, 422차원")
run("z_sum", "Eq(7) 합산, 211차원")
