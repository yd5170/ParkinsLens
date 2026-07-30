# -*- coding: utf-8 -*-
"""
final_resnet.py - 최종 확정 3D-ResNet 모델 (Model-2, Figure 3 사양).

[2026-07-27 분리] 원래 final_models.py 하나에 CNN3D 별칭과 ResNet3D가 같이 있었는데,
"final cnn과 final resnet으로 파일을 나눠달라"는 요청에 따라 분리. 최종 CNN은
final_cnn.py 참조.

현재 확정 체크포인트: checkpoints/resnet3d_20260722_170643_acc54.3.pt (원본,
dropout/augment 미적용) / checkpoints/resnet3d_20260727_153304_acc45.7.pt
(dropout=0.5 + augmentation + val_acc 체크포인트 기준, 과적합 대응 버전 -
Variant3_ResNet_CCA_WOA_결과요약.xlsx 참조)

[논문에 명시된 값 - 그대로 사용]
- 15 layers, Residual Block 5개(각 3 Residual Unit = 15 unit), Conv3D Block 2개(선행)
- FC-4: 1000개 특징 (본문 p.5 "1000 features are extracted from the 14th layer")
- Activation: ReLU, Eq.(1)

[논문에 기재되지 않아 자체 결정한 값]
- 채널 폭/필터 수: 논문 Figure 3에 전혀 수치가 없어(전 항목 "논문에 기재되지 않음"),
  표준 ResNet 채널 더블링 관행(64->128->256->512->512)을 자체 적용함. 커널 크기는
  3D-CNN과 동일하게 (3,3,3)로 통일.
"""
import torch
import torch.nn as nn


class ResidualUnit3D(nn.Module):
    """Figure 3 Residual Unit: main path(conv-bn-relu-conv-bn) + skip path(conv-bn) -> add -> relu"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResidualBlock3D(nn.Module):
    """Block = 3 Residual Unit (Figure 3: Block당 3 unit, 총 5 Block = 15 unit)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.unit0 = ResidualUnit3D(in_ch, out_ch, stride=1)
        self.unit1 = ResidualUnit3D(out_ch, out_ch, stride=1)
        self.unit2 = ResidualUnit3D(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.unit0(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class ResNet3D(nn.Module):
    """Model-2: Improved 3D-ResNet (Figure 3 사양, 15 layers = 5 Residual Block x 3 unit)

    [2026-07-26 추가] dropout - 파라미터 94,998,051개에 train 샘플이 212개뿐이라
    실측 과적합 확인됨(Train Acc 65.09% vs Val Acc 46.67%, Best Validation 시트 참조).
    Table5(논문)에 dropout 항목 자체가 없어 자유롭게 결정 가능한 영역 - FC-4와
    classifier 사이에 추가(가장 표준적인 위치). Dropout은 학습 파라미터가 없는
    레이어라 기존 체크포인트(dropout 없이 학습된 것) 로드에도 영향 없음."""

    def __init__(self, num_classes=3, in_channels=1, dropout=0.5):
        super().__init__()
        # Conv3D Block x2 (선행 stem, Figure 3 Stage 2-3)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # 5 Residual Block, 각 뒤에 MaxPool3D (Figure 3 순서 그대로)
        self.block0 = ResidualBlock3D(64, 64)
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)      # 56->28
        self.block1 = ResidualBlock3D(64, 128)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)      # 28->14
        self.block2 = ResidualBlock3D(128, 256)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)      # 14->7
        self.block3 = ResidualBlock3D(256, 512)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)      # 7->3
        self.block4 = ResidualBlock3D(512, 512)
        # [2026-07-22 정정] 논문 원문("five residual blocks with max-pooling layers")은
        # 5개 블록 전부 max-pooling이라고 명시하는데 여기만 AdaptiveAvgPool3d를 썼던 것을
        # 발견 - MaxPool3d(kernel=2,stride=2)로 정정(3->1, floor(3/2)=1로 산술상 문제없음).
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)      # 3->1

        # [2026-07-22 재정정] 바로 전 커밋에서 Figure3 다이어그램의 "FV-4 x 2000" 라벨을
        # 근거로 1000->2000으로 바꿨었는데, 이후 "Feature extraction" 본문과 Algorithm 1
        # 의사코드(Feature concatenation)를 함께 재검토한 결과 그 판단이 틀렸음이 드러남.
        # 본문: "1000 features were extracted [CNN FC1/FC2에서]... Additionally, 1000
        # features are extracted from the 14th layer of an improved 3D ResNet. A total of
        # 2000 features are finally extracted from BOTH architectures." - 여기서 "2000"은
        # CNN 1000 + ResNet 1000 = 합계 2000이라는 뜻이지 ResNet 혼자 2000이라는 뜻이
        # 아니었음(CCA 절의 "2000 features from both the ResNet and CNN modules"도 같은
        # 의미). Figure2/3의 "FV-3/4 x 2000" 다이어그램 라벨이 이 문맥에서 오히려
        # 오기(誤記)로 판단 - 원래 값(1000)으로 되돌림.
        self.fc4 = nn.Linear(512, 1000)          # FC-4: 1000 features (본문 "Feature extraction" 절 기준)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = torch.flatten(x, start_dim=1)

        fc4_feat = self.fc4(x)
        # [2026-07-26 추가] dropout은 classifier 입력 직전에만 적용 - fc4_feat 자체는
        # (04_Feature_Engineering의 CCA 융합 등에서) 원래 값 그대로 재사용해야 하므로
        # dropout 적용 전 값을 return_features로 반환.
        logits = self.classifier(self.dropout(fc4_feat))

        if return_features:
            return logits, {"fc4": fc4_feat}
        return logits


if __name__ == "__main__":
    resnet = ResNet3D()
    x = torch.randn(2, 1, 56, 56, 56)
    logits, feats = resnet(x, return_features=True)
    print("ResNet3D logits:", logits.shape, "| fc4:", feats["fc4"].shape)
