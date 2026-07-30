# -*- coding: utf-8 -*-
"""
inference.py - 09_Service의 Streamlit 앱(app.py)이 쓰는 추론 계층.

[실행 환경] 03_Model_Training/train_ablation.py와 동일하게 로컬 GPU/PyTorch
환경에서 실행하는 것을 전제로 한다. 개발 샌드박스는 torch DLL 로드 오류가 있어
여기서 직접 실행 검증은 못 했다.

[전제] 03_Model_Training/checkpoints/ 아래에 train_ablation.py가 저장한 .pt
체크포인트가 최소 1개 있어야 한다(체크포인트 저장 기능은 train_ablation.py에
이번에 추가됨 - 그 전 실행 결과에는 .pt가 없다).

[입력 데이터 범위 - 중요] 이 데모는 아직 임의의 원본 MRI를 업로드받아 전처리
(정합/두개골 제거/N4/리사이즈)하는 파이프라인을 갖추지 않았다. 01_Preprocessing이
그 자체로 여러 단계·여러 버전(v1/v2/z-score)을 가진 별도 파이프라인이라 여기서
재구현하지 않았고, 대신 이미 전처리 완료된 56^3 볼륨(01_Preprocessing의 표준
전처리 결과 폴더)에서 피험자를 선택하는 방식으로 "실제 모델 추론"을 보여준다.
새 원본 MRI 업로드 지원이 필요하면 01_Preprocessing 파이프라인을 그대로 호출하는
전처리 단계를 이 파일에 추가해야 한다.
"""
import csv
import os
import sys

import numpy as np
import torch
from scipy.ndimage import zoom

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _rel in ["02_Model_Definition", "03_Model_Training", "09_Service"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_models import CNN3D_Base, CNN3D_Variant1, CNN3D_Variant1_Untuned, CNN3D_Variant2, CNN3D_Variant3
from gradcam3d import GradCAM3D, get_target_layer

# [2026-07-26 추가] 최종 모델(Variant3, 클래스 붕괴 해결 버전) 데모 지원 추가.
# 체크포인트의 "variant" 필드가 "variant3"인 것을 로드하면 자동으로 이 클래스가 쓰임
# (train_ablation.py가 저장하는 필드명과 그대로 일치 - VARIANT_MODELS 키 이름 변경 금지).
VARIANT_MODELS = {
    "base": CNN3D_Base,
    "variant1": CNN3D_Variant1,
    "variant1_untuned": CNN3D_Variant1_Untuned,
    "variant2": CNN3D_Variant2,
    "variant3": CNN3D_Variant3,
}
CLASS_NAMES = ["Control", "Prodromal", "PD"]

CHECKPOINT_DIR = os.path.join(_ROOT, "03_Model_Training", "checkpoints")
# train_ablation.py의 기본값과 동일 - 01_Preprocessing/README.md 기준 표준 전처리 결과
DEFAULT_CSV = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
DEFAULT_IMAGE_DIR = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")


def list_checkpoints():
    """checkpoints/ 폴더의 .pt 파일 목록을 최신순으로 반환."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    return sorted(files, reverse=True)


def load_model(checkpoint_filename, device="cpu"):
    """checkpoints/<checkpoint_filename>을 로드해 (model, checkpoint_dict)를 반환."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    checkpoint = torch.load(ckpt_path, map_location=device)
    variant = checkpoint["variant"]
    model_cls = VARIANT_MODELS[variant]
    model = model_cls(num_classes=checkpoint.get("num_classes", 3)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_sample_list(csv_path=DEFAULT_CSV, image_dir=DEFAULT_IMAGE_DIR):
    """전처리된 볼륨 파일이 실제로 존재하는 샘플만 골라 반환(대시보드 선택 목록용)."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vol_path = os.path.join(image_dir, f"{row['sample_id']}.nii.gz")
            if os.path.exists(vol_path):
                rows.append(row)
    return rows


def load_volume(sample_id, image_dir=DEFAULT_IMAGE_DIR):
    """전처리 완료된 56^3 T2 볼륨을 로드한다. nibabel은 로컬 환경에 설치돼 있어야 한다."""
    import nibabel as nib
    path = os.path.join(image_dir, f"{sample_id}.nii.gz")
    return nib.load(path).get_fdata(dtype=np.float32)


def predict_with_cam(model, volume, device="cpu"):
    """모델 forward + Grad-CAM을 한 번에 수행.

    반환 dict:
      - probs: {"Control": p0, "Prodromal": p1, "PD": p2}
      - pred_idx / pred_label
      - cam: 원본 볼륨과 같은 shape(56,56,56)으로 리사이즈된 0~1 히트맵
    """
    model.eval()
    x = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

    target_layer = get_target_layer(model)
    with GradCAM3D(model, target_layer) as cam_engine:
        cam, probs, pred_idx = cam_engine(x)

    zoom_factors = [s / c for s, c in zip(volume.shape, cam.shape)]
    cam_full = zoom(cam, zoom_factors, order=1)
    cam_full = np.clip(cam_full, 0.0, 1.0)

    return {
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "pred_idx": pred_idx,
        "pred_label": CLASS_NAMES[pred_idx],
        "cam": cam_full,
    }
