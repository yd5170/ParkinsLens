"""전처리 단계별(BET/정합/정규화/최종 리사이즈) 육안 확인용 QC 스크립트.

qc_check.py가 최종 결과물에 대한 수치 QC(NaN/범위 등)라면, 이 스크립트는
Control/Prodromal/PD 각 1명씩 뽑아 BET->정합->정규화->최종(56^3) 4단계를
sagittal/coronal/axial 중앙 슬라이스로 나란히 그려서 눈으로 이상 유무를
확인하기 위한 것. 대상 피험자는 _work/ 하위에 중간 단계 파일이 남아있는
샘플 중에서 고름(중간 파일을 지운 샘플은 최종본만 남아있어 비교 불가).
"""
import os
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_SCRIPT_DIR, "..", "전처리_ref21order_v1")
OUT_PATH = os.path.join(_SCRIPT_DIR, "..", "QC_전처리단계_시각화_3샘플.png")

# (sample_id, 라벨) - data_0713_wsl_v2.csv 기준, _work/ 하위에 중간단계 파일이
# 남아있는 샘플 중 각 그룹 1명씩
SAMPLES = [
    ("sub-3000_I224561", "Control"),
    ("sub-16785_I814554", "Prodromal"),
    ("sub-3001_I224572", "PD"),
]

STAGES = [
    ("01_bet", "BET(두개골제거) 직후"),
    ("02_reg", "정합(registration) 직후"),
    ("03_norm", "min-max 정규화 직후"),
    ("final", "최종(리사이즈 56^3)"),
]


def mid_slices(vol):
    x, y, z = vol.shape
    return vol[x // 2, :, :], vol[:, y // 2, :], vol[:, :, z // 2]


def main():
    fig, axes = plt.subplots(
        len(SAMPLES) * 3, len(STAGES),
        figsize=(4 * len(STAGES), 3 * len(SAMPLES) * 3),
    )

    for si, (sub_id, label) in enumerate(SAMPLES):
        for ti, (stage_key, stage_label) in enumerate(STAGES):
            if stage_key == "final":
                path = os.path.join(ROOT, f"{sub_id}.nii.gz")
            else:
                path = os.path.join(ROOT, "_work", sub_id, f"{sub_id}_{stage_key}.nii.gz")
            vol = nib.load(path).get_fdata()
            sag, cor, ax = mid_slices(vol)
            row_base = si * 3
            for pi, (sl, plane) in enumerate([(sag, "Sagittal"), (cor, "Coronal"), (ax, "Axial")]):
                axp = axes[row_base + pi, ti]
                axp.imshow(np.rot90(sl), cmap="gray")
                axp.axis("off")
                if pi == 0:
                    axp.set_title(f"{label}({sub_id})\n{stage_label}\nshape={vol.shape}", fontsize=8)
                else:
                    axp.set_title(plane, fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
