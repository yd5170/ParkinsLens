import os
import sys
import json
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

_ROOT = r"D:\new tensor"
sys.path.insert(0, os.path.join(_ROOT, "02_Model_Definition"))
sys.path.insert(0, os.path.join(_ROOT, "03_Model_Training"))

from ablation_models import CNN3D_Base, CNN3D_Variant1, CNN3D_Variant1_Untuned, CNN3D_Variant2, CNN3D_Variant3
from final_resnet import ResNet3D
from cnn3d_figure2 import CNN3D_Figure2 as CNN3D_Figure2Legacy
from train_ablation import build_loaders

VARIANT_MODELS = {
    "base": CNN3D_Base,
    "variant1": CNN3D_Variant1,
    "variant1_untuned": CNN3D_Variant1_Untuned,
    "variant2": CNN3D_Variant2,
    "variant3": CNN3D_Variant3,
}
# [2026-07-23 추가] ResNet3D는 ablation 5개와 생성자 시그니처가 다름
# (input_size 인자가 없음, 체크포인트에도 "input_size" 키가 저장되지 않음) -
# 별도 분기로 처리.
# [2026-07-24 추가] 1차/Figure2 모델도 동일한 이유로 별도 분기(input_size 인자
# 없음 - ablation_models.py의 CNN3D_* 클래스들과는 다른 파일의 별개 클래스).
# [2026-07-26 파일 분리, 2026-07-27 파일명 갱신] 원래 models.py 하나에 1차 모델과
# 최종 모델이 같이 있었는데, cnn3d_figure2.py(1차)/final_cnn.py+final_resnet.py
# (최종=CNN3D_Variant3 별칭 / ResNet3D)로 분리됨.
# ckpt["variant"]=="cnn3d"(1차/Figure2 모델 체크포인트)는 cnn3d_figure2.CNN3D_Figure2로
# 불러와야 함 - final_cnn.CNN3D(=Variant3)로 불러오면 shape mismatch가 남.


def analyze_split(ckpt_path, split="test"):
    """[2026-07-24 추가] split={"train","val","test"} 중 하나를 골라 그 세트에 대한
    accuracy/precision/recall/f1(macro+weighted)/confusion matrix/클래스별 예측개수를
    계산. 기존 analyze()(test 전용)를 일반화한 버전 - Best Validation 시트의
    "없음"이던 val/train 지표들을 재학습 없이 저장된 체크포인트(best-epoch 가중치)로
    다시 추론해서 채우는 데 사용."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split은 train/val/test 중 하나여야 합니다 (요청값: {split})")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ckpt["variant"]
    hp = ckpt["hyperparams"]
    class_names = ckpt["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if variant == "resnet3d":
        model = ResNet3D(num_classes=ckpt["num_classes"]).to(device)
    elif variant == "cnn3d":
        model = CNN3D_Figure2Legacy(num_classes=ckpt["num_classes"]).to(device)
    else:
        model_cls = VARIANT_MODELS[variant]
        model = model_cls(num_classes=ckpt["num_classes"], input_size=ckpt["input_size"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    csv_path = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
    image_dir = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")
    train_loader, val_loader, test_loader = build_loaders(csv_path, image_dir, hp["batch_size"], hp["seed"])
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    pred_counts = {class_names[i]: all_preds.count(i) for i in range(len(class_names))}

    result = {
        "variant": variant,
        "checkpoint": ckpt_path,
        "split": split,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f_weighted,
        "pred_counts": pred_counts,
    }
    return result


def analyze(ckpt_path):
    """기존 호출부(build_report.py 등) 호환용 - test split 분석."""
    return analyze_split(ckpt_path, split="test")


if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = analyze(ckpt_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
