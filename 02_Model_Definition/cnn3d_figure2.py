# -*- coding: utf-8 -*-
"""
cnn3d_figure2.py - 1차 모델(Stage 1): 3D-CNN, Figure 2 사양을 문자 그대로 재현.

[2026-07-26 분리] 원래 models.py 안에 CNN3D_Figure2Legacy라는 이름으로 최종 모델
(ablation_models.py의 CNN3D_Variant3)과 한 파일에 같이 있었는데, "1차 모델은 별도
파일로 분리해 저장해달라"는 요청에 따라 이 파일로 옮김. 최종 모델은
final_cnn.py(CNN3D=CNN3D_Variant3 별칭)와 final_resnet.py(ResNet3D) 참조.

근거 문서: 05_Document/모델_아키텍처_분석.md (Priyadharshini et al., Sci Rep 2024, 14:23394,
Figure 2 전수 검토 결과)

[논문에 명시된 값 - 그대로 사용]
- 3D-CNN 필터 진행: 32-64-128-[pool]-[bn]-256-512-1024-[pool]-[bn]-512-256-[pool]-[bn]
- 3D-CNN 커널 크기: (3,3,3)
- 3D-CNN MaxPool size: (2,2,2), 총 3회
- 3D-CNN Flatten 차원: 12544 (Figure 2 원문 표기)
- 3D-CNN FC-1/FC-2: 각 1000개 특징 (본문 p.5)
- Activation: ReLU, Eq.(1)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 참조]
- Stride, Padding (Conv3D 전 레이어): stride=1, padding=1 (SAME) 사용.
  근거: Figure 2의 단계별 출력 크기(56->56->56->28(pool)->...)가 pool 전에는
  공간크기가 변하지 않는 것으로 표기되어 있어, conv 자체는 크기를 보존해야 함.
  이를 만족하는 유일한 표준 조합이 stride=1, padding=1(3x3x3 커널 기준)이므로 채택.
- Flatten=12544를 만들기 위한 방법(depth_collapse): 최종 conv 출력(7,7,7,256)에서
  depth(D) 축만 AdaptiveAvgPool3d로 1로 축소 -> (1,7,7,256) -> flatten 12544.
  논문 자체의 산술 불일치(7*7*7*256=87808 != 12544)를 그대로 두는 대신,
  Figure 2에 명시된 최종 수치(12544)를 재현 목표로 우선시한 해석적 결정.
- FV-3(3D-CNN 내부 FC-1+FC-2 융합) 방식: concatenation 채택(2000차원).
  본문 Eq.2는 elementwise max로 서술되어 있으나 그 경우 결과 차원이 1000이 되어
  Figure 1의 "FV-3 x2000" 라벨과 모순되므로, 최종 차원이 2000이 되는 concatenation을
  채택하고 이유를 명시함(모델_아키텍처_분석.md 불일치사항 3 참조).

[2026-07-23 추가] 중요 - ablation_models.py의 CNN3D_Variant3와 이 클래스의 차이.

둘 다 "24-layer 최종 모델"을 가리키지만 서로 다른 해석으로 재구현되어 있어 실제
구현이 다름(사용자 제공 정리표 기준, C:\\Users\\user\\Pictures\\figure2add.png):

| 항목                  | Figure2 실제(이 클래스, CNN3D_Figure2) | ablation_models.py CNN3D_Variant3      |
|-----------------------|------------------------------------------|------------------------------------------|
| Conv 채널 진행        | 32-64-128-256-512-1024-512-256           | 32-64 / 128-256-512 / 512-256-128        |
| Flatten 방식          | 직접 Flatten = 12544 (depth_collapse)    | Global Max Pooling -> 256                |
| FC-1/FC-2 입력 차원   | 12544                                     | 256                                       |

이 클래스는 논문 Figure 2 다이어그램의 채널 수/Flatten 차원(12544)을 문자 그대로
재현하는 쪽을 택했고, CNN3D_Variant3는 논문 Table 3 Study 2가 Variant3에 대해 명시한
그리드서치 결과("Flatten=Global max")를 재현하는 쪽을 택함 - 두 파일 모두 각자 근거
(다이어그램 vs 표)가 있는 별개의 해석이며, 어느 한쪽이 "틀린" 것이 아님. 두 클래스를
섞어 쓰지 말 것(예: 이 클래스의 체크포인트를 CNN3D_Variant3 구조로 로드하면 shape
mismatch 발생). 상세 근거는 DEVIATIONS.md 1번/9번 섹션 참조.

체크포인트: checkpoints/cnn3d_20260724_015432_acc45.7.pt (train_cnn3d.py로 학습,
Table3 Study2 그리드서치 대상이 아니라서 논문 비교 목표치 없음).
"""
import torch
import torch.nn as nn


class CNN3D_Figure2(nn.Module):
    """1차 모델(Stage 1): 3D-CNN (Figure 2 사양 문자 그대로 재현). 파일 상단 docstring 참조."""

    def __init__(self, num_classes=3, in_channels=1, flatten_mode="depth_collapse"):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)   # Input Layer, Filter=32
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)            # Conv3D-1, Filter=64
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)           # Conv3D-2, Filter=128
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 56->28
        self.bn1 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)          # Conv3D-3, Filter=256
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)          # Conv3D-4, Filter=512
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1)         # Conv3D-5, Filter=1024
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 28->14
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)         # Conv3D-6, Filter=512
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)          # Conv3D-7, Filter=256
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)                            # MaxPool: 14->7
        self.bn3 = nn.BatchNorm3d(256)

        self.relu = nn.ReLU(inplace=True)

        self.flatten_mode = flatten_mode
        if flatten_mode == "depth_collapse":
            self.spatial_collapse = nn.AdaptiveAvgPool3d((1, 7, 7))
            flatten_dim = 1 * 7 * 7 * 256   # = 12544, 논문 Figure 2 명시값과 일치
        elif flatten_mode == "full":
            self.spatial_collapse = nn.Identity()
            flatten_dim = 7 * 7 * 7 * 256   # = 87808, 산술적으로 정확하나 논문 표기와 불일치
        else:
            raise ValueError(f"Unknown flatten_mode: {flatten_mode}")

        self.flatten_dim = flatten_dim
        self.fc1 = nn.Linear(flatten_dim, 1000)   # FC-1: 1000 features (본문 p.5)
        self.fc2 = nn.Linear(flatten_dim, 1000)   # FC-2: 1000 features (본문 p.5)
        self.classifier = nn.Linear(2000, num_classes)  # FV-3(2000) -> Softmax(3)

    def forward(self, x, return_features=False):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.spatial_collapse(x)
        x = torch.flatten(x, start_dim=1)

        fc1_feat = self.fc1(x)
        fc2_feat = self.fc2(x)
        fv3 = torch.cat([fc1_feat, fc2_feat], dim=1)

        logits = self.classifier(fv3)
        if return_features:
            return logits, {"fc1": fc1_feat, "fc2": fc2_feat, "fv3": fv3}
        return logits


if __name__ == "__main__":
    cnn = CNN3D_Figure2()
    x = torch.randn(2, 1, 56, 56, 56)
    logits, feats = cnn(x, return_features=True)
    print("CNN3D_Figure2(1차) logits:", logits.shape, "| fv3:", feats["fv3"].shape, "| flatten_dim:", cnn.flatten_dim)
