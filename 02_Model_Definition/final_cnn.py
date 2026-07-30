# -*- coding: utf-8 -*-
"""
final_cnn.py - 최종 확정 CNN 모델: CNN3D(= ablation_models.py의 CNN3D_Variant3 별칭).

[2026-07-27 분리] 원래 final_models.py 하나에 CNN3D 별칭과 ResNet3D가 같이 있었는데,
"final cnn과 final resnet으로 파일을 나눠달라"는 요청에 따라 분리. ResNet3D는
final_resnet.py 참조.

[2026-07-26 정리] 이 프로젝트의 3D-CNN 파이프라인 단계:
  1차 모델(Stage 1) = cnn3d_figure2.py의 CNN3D_Figure2 (Figure2 다이어그램 문자 그대로 재현)
  최종 모델(Stage 2/확정) = ablation_models.py의 CNN3D_Variant3
    (Table3 Study2 그리드서치 확정 구성 - Pooling=Average, Flatten=Global max,
     FC-1/FC-2 병렬 분기 - 논문 재현 충실도가 가장 높은 버전으로 확정됨,
     상세 비교는 02_Model_Definition/README.md 참조)

클래스 본체를 여기 복제하지 않고 import 별칭만 두는 이유: 코드가 두 곳에 있으면
한쪽만 고치고 잊어버리는 사고가 나기 쉬움(이 프로젝트에서 실제로 몇 번 겪은 문제) -
구조의 단일 소스는 항상 ablation_models.py 유지.

현재 확정 체크포인트: checkpoints/ablation_variant3_20260726_035445_acc50.0.pt
(class-weighted loss + WeightedRandomSampler + val_acc 체크포인트 기준으로 재학습해
클래스 붕괴 해결 - Test Acc 50.00%, Control/Prodromal/PD recall=65%/38%/43%)
"""
import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from ablation_models import CNN3D_Variant3

# ============================================================
# 최종 모델(CNN3D) - ablation_models.py의 CNN3D_Variant3를 그대로 재노출
# ============================================================
CNN3D = CNN3D_Variant3


if __name__ == "__main__":
    cnn = CNN3D()  # = CNN3D_Variant3
    x = torch.randn(2, 1, 56, 56, 56)
    logits, feats = cnn(x, return_features=True)
    print("CNN3D(최종=Variant3) logits:", logits.shape, "| fv3:", feats["fv3"].shape, "| gmp_dim:", cnn.gmp_dim)
