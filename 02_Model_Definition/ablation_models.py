# -*- coding: utf-8 -*-
"""
ablation_models.py - 논문 Ablation Study(Table 3, Study 1) 재현용 3D-CNN 변형 모델

근거: 주논문_nature.pdf 본문 "Ablation study" 절(502~516행, pdftotext 추출 기준),
07_Document/모델_아키텍처_분석.md Table 3 인용부.

논문은 최종 채택 모델(Variant 3, 24-layer, models.py의 CNN3D)에 도달하기까지
4단계 ablation 실험을 수행했습니다. 본 파일은 그 중 Base/Variant1/Variant2를
단계적으로 구현합니다(사용자 요청에 따라 Base부터 순서대로 작성).
[2026-07-20: Variant1(CNN3D_Variant1, 9-layer), Variant2(CNN3D_Variant2, 17-layer) 구현 완료 - 아래 참조]

[논문 원문 그대로 인용 - Base 모델, Stage 1]
"In stage 1, the base model was designed with 8 layers, starting with the input
layer, followed by two 3D-convolutional (3D-conv) layers, an activation layer,
a pooling layer, a Batch Normalization (BN) layer, a flattened layer, and a
fully connected SoftMax layer, which resulted in a model accuracy of 82.02%."

8-layer 구성 (원문 그대로):
  1. Input
  2-3. Conv3D x2
  4. Activation (ReLU)
  5. Pooling (MaxPool3D)
  6. BatchNorm3D
  7. Flatten
  8. Fully Connected + SoftMax(classifier)

[논문에 기재되지 않아 프로젝트에서 자체 결정한 값 - DEVIATIONS.md 반영 예정]
- Conv 채널 폭(filter 수): 논문은 Base 모델의 채널 수를 명시하지 않음. 최종 채택
  모델(Variant 3, models.py CNN3D)의 첫 conv 진행이 32->64->128이므로, 그 앞부분
  2개 conv와 동일한 채널(32->64)을 사용해 변형 간 계열 일관성을 유지함.
- Conv stride/padding: models.py와 동일하게 stride=1, padding=1(3x3x3 커널, SAME)
  사용 - 공간 크기를 conv 전후로 보존하기 위함(Figure 2 해석과 동일 근거).
- Pooling: kernel=2, stride=2 (56 -> 28). 논문 "pooling layer"는 종류(Max/Average)를
  Table 3 Study 2에서 그리드 서치 대상으로 명시했을 뿐 Base 모델 자체의 선택은
  기재하지 않음. 최종 채택 모델과 동일하게 MaxPool3d를 기본값으로 사용.
- Flatten 이후 별도의 은닉 Dense(1000차원 FC-1/FC-2) 없이 곧바로 분류기로 연결.
  논문이 "flattened layer, and a fully connected SoftMax layer"라고 명시해 은닉
  Dense를 언급하지 않았으므로, 원문 그대로 Flatten -> Linear(num_classes) 구조로
  구현함 (Variant 3의 FC-1/FC-2/2000차원 FV-3 구조는 이 Base 모델에는 해당 없음).
"""
import torch
import torch.nn as nn


class CNN3D_Base(nn.Module):
    """Ablation Study Stage 1 - Base 모델 (8-layer, 논문 보고 정확도 82.02%)

    구조: Conv3D(1->32) -> Conv3D(32->64) -> ReLU -> MaxPool3D(2) -> BatchNorm3D
          -> Flatten -> Linear(-> num_classes)

    입력: (N, 1, 56, 56, 56)  (전처리06_리사이즈_최종 규격)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        # 레이어 2-3: Conv3D x2 (채널 32 -> 64), 공간 크기 보존(stride=1, padding=1)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)

        # 레이어 4: Activation (ReLU) - 두 conv 다음에 한 번만 적용
        # (논문이 "two 3D-conv layers, an activation layer"로 단수 표기한 것을
        #  그대로 따름 - conv1/conv2를 거친 뒤 activation을 1회 적용)
        self.relu = nn.ReLU(inplace=True)

        # 레이어 5: Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # 56 -> 28

        # 레이어 6: BatchNorm3D
        self.bn = nn.BatchNorm3d(64)

        # 레이어 7: Flatten
        pooled_size = input_size // 2  # 56 -> 28
        self.flatten_dim = 64 * pooled_size * pooled_size * pooled_size  # 64*28*28*28 = 1,404,928
        self.flatten = nn.Flatten()

        # 레이어 8: Fully Connected + SoftMax 분류기
        # (SoftMax 자체는 nn.CrossEntropyLoss에 내장되어 있으므로 forward는 logits만 반환)
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU 계열 활성화에 적합) - models.py와 동일한 프로젝트 관행"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


class CNN3D_Variant1(nn.Module):
    """Ablation Study Stage 2 - Variant 1 (9-layer, 논문 보고 정확도 85.75%)

    [논문 원문 그대로 인용 - Variant 1, Stage 2]
    "In stage 2, a slightly modified version of the original base model, variant
    1 architecture was designed with a 9-layer architecture, with an input layer,
    three 3D-conv layers, an activation layer, a pooling layer, BN layer, a
    flattened dense layer, and a SoftMax layer, achieving a model accuracy of
    85.75%."

    9-layer 구성 (원문 그대로):
      1. Input
      2-4. Conv3D x3 (Base보다 conv 1개 더 있음 - 유일한 구조 차이)
      5. Activation (ReLU)
      6. Pooling (MaxPool3D)
      7. BatchNorm3D
      8. Flatten
      9. Fully Connected + SoftMax(classifier)

    Base와의 구조적 차이(Study1)는 conv 레이어 개수(2->3)뿐 - Table 3 Study 1 표
    (Conv layer 수만 2->3으로 변화, Pooling layer 수는 1로 동일)와 정확히 일치.

    [2026-07-20 갱신: Table 3 Study 2 표에서 Variant1 열의 체크 표시를 직접(육안)
    확인한 결과 - Base와 다른 조합이 채택돼 있었음이 확인됨. 기존엔 Base와 동일
    조합(Pooling=Max, Flatten=Flatten, Batch=32, LR=0.01)으로 잠정 가정하고
    있었으나 전부 오판이었음 - 아래 값으로 정정]
    - **Pooling = Average** (Base는 Max) -> AvgPool3d로 변경
    - **Flatten = Global max** (Base는 Flatten) -> 이 값은 단순 하이퍼파라미터가
      아니라 구조 자체가 바뀜: conv 결과를 통째로 펼치는 대신, 채널별로 공간
      전체에서 최댓값 하나만 남기는 Global Max Pooling(AdaptiveMaxPool3d(1))을
      적용 -> classifier 입력이 2,809,856(128*28^3) 대신 **128**(채널 수)로 줄어듦.
      이전 버전은 classifier가 모델 파라미터의 98.7%를 차지해 학습 불안정성
      (클래스 붕괴)의 핵심 원인이었는데, Global max를 쓰면 classifier가
      128->3(387개 파라미터)로 작아져 이 문제 자체가 원천적으로 다를 수 있음
      (03_Model_Training/클래스_붕괴_분석_및_대응.md 참조 - 그 분석은 이 잘못된
      구조를 대상으로 한 것이었음)
    - Activation = ReLU(Base와 동일), Batch size = **64**(Base는 32),
      Learning rate = **0.001**(Base는 0.01) - train_ablation.py의 하이퍼파라미터
      비교표에도 반영됨

    [논문에 기재되지 않아 프로젝트에서 자체 결정한 값]
    - Conv 채널 폭: Base(32->64)에 이어 세 번째 conv를 64->128로 확장 - 최종 채택
      모델(Variant 3, models.py CNN3D)의 32->64->128 진행과 일치시켜 계열 일관성 유지
      (Base의 32->64 결정과 동일한 근거)
    - Conv stride/padding 등 나머지 자체 결정 사항은 Base와 동일 근거 적용(위
      CNN3D_Base docstring 참조)

    입력: (N, 1, 56, 56, 56)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        # 레이어 2-4: Conv3D x3 (채널 32 -> 64 -> 128), 공간 크기 보존(stride=1, padding=1)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        # 레이어 5: Activation (ReLU) - Base와 동일하게 conv 3개 다음 1회 적용
        # (논문이 "three 3D-conv layers, an activation layer"로 단수 표기)
        self.relu = nn.ReLU(inplace=True)

        # 레이어 6: Pooling - Table3 Study2 Variant1 열 확인 결과 Average(Base는 Max)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)  # 56 -> 28

        # 레이어 7: BatchNorm3D
        self.bn = nn.BatchNorm3d(128)

        # 레이어 8: Flatten -> Table3 Study2 Variant1 열 확인 결과 Global max
        # (단순 Flatten이 아니라 채널별 전역 최댓값 - classifier 입력이 128차원으로 축소)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.flatten_dim = 128  # Global max pooling 이후 채널 수만 남음(공간 정보는 소거)

        # 레이어 9: Fully Connected + SoftMax 분류기
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU 계열 활성화에 적합) - CNN3D_Base와 동일한 프로젝트 관행"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


class CNN3D_Variant1_Untuned(nn.Module):
    """Ablation Study Stage 2 - Variant 1의 "튜닝 전"(Study 1) 버전.

    [2026-07-21 신설 배경] 논문 Table 3는 Study1(구조 탐색, 하이퍼파라미터 미기재)과
    Study2(구조 고정 후 하이퍼파라미터 그리드서치)를 구분해서 보고함. 우리도 이 구분을
    재현하기 위해, "튜닝 전" 실험은 Base/Variant1/Variant2/Variant3 전부 공통 기본
    하이퍼파라미터(Pooling=Max, Flatten=Flatten, Batch=64, LR=0.01, Adam, ReLU)로
    통일해서 돌리기로 함(사용자 결정 - 논문이 Study1 하이퍼파라미터를 명시하지 않아
    자체 결정한 값). Base와 Variant2는 마침 이 공통값이 Study2 확정값과 동일해서
    별도 클래스가 필요 없지만, Variant1은 Study2에서 Average pooling + Global max
    로 바뀌므로 구조가 실제로 달라짐 - 그래서 이 클래스를 별도로 둠.

    conv 레이어 구성(3개, 채널 32->64->128)과 activation 해석(단수, conv 3개 후
    1회)은 CNN3D_Variant1과 동일 - 오직 Pooling(Average->Max)과 Flatten(Global
    max->일반 Flatten)만 다름. 이 차이 때문에 classifier 입력이 128(Global max)
    대신 2,809,856(128*28^3, 일반 Flatten)으로 커짐 - CNN3D_Variant1의 "정정 전"
    구현과 사실상 동일한 구조.

    입력: (N, 1, 56, 56, 56)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # 56 -> 28 (공통 기본값: Max)
        self.bn = nn.BatchNorm3d(128)

        pooled_size = input_size // 2
        self.flatten_dim = 128 * pooled_size * pooled_size * pooled_size  # 128*28^3 = 2,809,856
        self.flatten = nn.Flatten()  # 공통 기본값: 일반 Flatten(Global max 아님)
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


class CNN3D_Variant2(nn.Module):
    """Ablation Study Stage 3 - Variant 2 (17-layer, 논문 보고 정확도 88.76%)

    [논문 원문 그대로 인용 - Variant 2, Stage 3]
    "In stage -3, Variant 2 was designed with a 17-layer architecture, including
    an input layer, two 3D-conv layers, activation layers, a pooling layer, a BN
    layer, three 3D-conv layers, activation layers, a pooling layer, a BN layer,
    a flatten dense layer, and a SoftMax layer, achieving an accuracy of 88.76%."

    [2026-07-20 구조 해석 근거] Base/Variant1은 "an activation layer"(단수)라고
    썼지만 Variant2부터는 "activation layer**s**"(복수)로 바뀜. 레이어 수를 직접
    맞춰본 결과:
      - 단수 해석(블록당 activation 1개, Base/Variant1 방식) -> 14층, 17층과 불일치
      - **복수 해석(conv마다 activation 1개, 표준 conv-ReLU-conv-ReLU 패턴)**
        -> 1(input) + [2conv+2act+1pool+1BN=6] + [3conv+3act+1pool+1BN=8]
           + 1(flatten) + 1(softmax) = **정확히 17** (일치)
    복수 해석이 논문이 명시한 17층과 정확히 일치하므로 이를 채택. 즉 Variant2부터는
    "conv 블록마다 conv 다음에 매번 ReLU를 적용"하는 표준 방식으로 구조가 전환됨
    (참고: 같은 방식으로 Variant3, 24층도 계산하면 23층이 나와 1층 차이가 남 -
    주논문 자체의 또 다른 내부 불일치로 판단, DEVIATIONS.md에 기재된 다른
    수치 불일치들과 같은 성격. Variant2 채택 근거와는 무관).

    17-layer 구성:
      1. Input
      2-3. Conv3D x2 (블록1, 채널 1->32->64)
      4-5. Activation x2 (conv마다 1개씩)
      6. Pooling (MaxPool3D)
      7. BatchNorm3D
      8-10. Conv3D x3 (블록2, 채널 64->128->256->512)
      11-13. Activation x3 (conv마다 1개씩)
      14. Pooling (MaxPool3D)
      15. BatchNorm3D
      16. Flatten
      17. Fully Connected + SoftMax(classifier)

    [논문에 기재되지 않아 프로젝트에서 자체 결정한 값]
    - Conv 채널 폭: 최종 채택 모델(Variant 3, models.py CNN3D)의 채널 진행
      32->64->128->256->512->...를 그대로 이어감(Base/Variant1과 동일한 근거) -
      블록1(conv 2개)은 Base와 동일한 32->64, 블록2(conv 3개)는 이어서 128->256->512
    - Pooling: 블록마다 kernel=2, stride=2 (56->28->14) - Base/Variant1과 동일 근거
    - Conv stride/padding, Flatten 이후 은닉 Dense 없음 등 나머지는 Base와 동일 근거

    입력: (N, 1, 56, 56, 56)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        # 블록 1: Conv3D x2 (채널 1 -> 32 -> 64), conv마다 ReLU
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 56 -> 28
        self.bn1 = nn.BatchNorm3d(64)

        # 블록 2: Conv3D x3 (채널 64 -> 128 -> 256 -> 512), conv마다 ReLU
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 28 -> 14
        self.bn2 = nn.BatchNorm3d(512)

        self.relu = nn.ReLU(inplace=True)  # 파라미터 없는 stateless 모듈 - 재사용해도 동작 동일

        pooled_size = input_size // 4  # 56 -> 28 -> 14 (풀링 2회)
        self.flatten_dim = 512 * pooled_size * pooled_size * pooled_size  # 512*14*14*14 = 1,404,928
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU 계열 활성화에 적합) - CNN3D_Base와 동일한 프로젝트 관행"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


class CNN3D_Variant3(nn.Module):
    """Ablation Study Stage 4 - Variant 3 (24-layer, 논문 보고 정확도 89.43%/93.41%)

    [논문 원문 그대로 인용 - Variant 3, Stage 4]
    "Variant 3 comprises a 24-layer architecture, starting with an input layer, two
    3D-conv layers, activation layers, a pooling layer, a BN layer, three 3D-conv
    layers, activation layers, a pooling layer, BN layer, two 3D-conv layers,
    activation layers, a pooling layer, a BN layer, and finally a flatten dense
    layer followed by a SoftMax layer, achieving the highest accuracy of 89.43%."

    [2026-07-24 재확정 - "input layer"와 "two/three/two 3D-conv layers"는 별개 항목]
    원문을 다시 쪼개보면 "an input layer"와 "two 3D-conv layers"가 콤마로 분리된
    별개 절임 - input layer 자체가 conv 1개(1->32)이고, 뒤따르는 "two/three/two"는
    **input을 제외한** 추가 conv 개수. 이렇게 셀 때만 아래 두 가지가 모두 정확히
    맞아떨어짐:

    1) conv 개수: input(1개, 1->32) + 블록1 "two"(2개, 32->64->128) + 블록2
       "three"(3개, 128->256->512->1024) + 블록3 "two"(2개, 1024->512->256)
       = 실제 conv 총 8개, Table3 Study1의 "two/three/two"(=7, input 제외)와
       정확히 일치. 실제 conv 8개는 Figure2 다이어그램의 필터 수 나열
       (32,64,128,256,512,1024,512,256 - 8개)과도 정확히 일치함.
    2) 총 레이어 24: input(1) + [블록1: conv2+활성화2+pool1+BN1=6] +
       [블록2: conv3+활성화3+pool1+BN1=8] + [블록3: conv2+활성화2+pool1+BN1=6] +
       FC-1(1) + FC-2(1) + SoftMax(1) = 1+6+8+6+1+1+1 = **정확히 24**.
       단, input layer(conv1) 자체에는 활성화를 적용하지 않음(Figure2의 Input
       Layer 박스에는 "+ReLU" 표기가 없고 Conv3D-1/2 박스에만 있음 - 이 조건이
       빠지면 25가 돼서 24와 안 맞음).

    [2026-07-23~24 판단 변경 이력] 이전 버전(~2026-07-23)은 "input layer"를
    비계산적 개념 항목으로 보고 conv1을 "두 개의 3D-conv layer" 중 첫 번째로
    셌음(블록1=conv1,conv2 2개, 채널 32->64에서 멈춤, 블록3=512->256->128).
    이 경우도 conv 총 7개(2+3+2)로 "two/three/two"와 표면적으로는 맞았으나,
    Figure2 다이어그램의 채널 진행(32-64-128-256-512-1024-512-256, 8개 필터
    값)과는 안 맞았음(7개 conv로는 8개 채널값을 다 못 담음). "input layer"를
    별개 conv로 분리하는 이번 해석이 원문 문장 구조(콤마로 구분된 병렬절)에도
    더 부합하고 Figure2 채널 수와도 완전히 일치하므로 이 해석으로 교체함.
    (사용자가 C:\\Users\\user\\Downloads\\ablation_models (1).py로 제공, 2026-07-24)

    [남은 비일관성] CNN3D_Base/Variant1/Variant2는 아직 이전 관례(input=conv1
    자체, 별도 분리 안 함)를 씀 - 통일 여부는 추후 결정.

    [Table 3 Study 2 그리드서치 확정 하이퍼파라미터 - 논문 표 육안 확인]
    Pooling=Average, Activation=ReLU, Batch size=32(Base/V1/V2는 64), Flatten=Global
    max, Optimizer=Adam, Learning rate=0.001, Epoch=30
    (정확도 93.41%, Precision 95.76%, Recall 88.98%, F1 91.68%)

    ⚠️ 2026-07-24 구조 변경으로 이전 체크포인트(채널 최대 512, gmp_dim=128)와
    shape이 달라짐 - 기존 체크포인트 재사용 불가, 재학습 필요.

    입력: (N, 1, 56, 56, 56)
    """

    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        # Input Layer(1->32) + 블록 1: Conv3D x2(32->64->128), Figure2 Input+Conv3D-1,2와 일치
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)   # Input Layer(활성화 없음)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)  # 56 -> 28 (Study2: Average)
        self.bn1 = nn.BatchNorm3d(128)

        # 블록 2: Conv3D x3 (채널 128 -> 256 -> 512 -> 1024), Figure2 Conv3D-3,4,5와 일치
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)  # 28 -> 14
        self.bn2 = nn.BatchNorm3d(1024)

        # 블록 3: Conv3D x2 (채널 1024 -> 512 -> 256), Figure2 Conv3D-6,7과 일치
        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)  # 14 -> 7
        self.bn3 = nn.BatchNorm3d(256)

        self.relu = nn.ReLU(inplace=True)

        # Flatten -> Study2 확정값(Global max)에 따라 AdaptiveMaxPool3d(1) 사용
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.gmp_dim = 256  # 블록3 최종 채널 수(256)만 남음

        # FC-1, FC-2(각 1000) 병렬 -> concat(2000) -> SoftMax 분류기 (다이어그램 그대로)
        self.fc1 = nn.Linear(self.gmp_dim, 1000)
        self.fc2 = nn.Linear(self.gmp_dim, 1000)
        self.classifier = nn.Linear(2000, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU 계열 활성화에 적합) - CNN3D_Base와 동일한 프로젝트 관행"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        """[2026-07-24 추가] return_features=True로 호출하면 (logits, {"fc1":...,
        "fc2":..., "fv3":...})를 반환 - models.py의 CNN3D/ResNet3D와 동일한 관례.
        04_Feature_Engineering/extract_features.py가 CCA용 FC-1/FC-2를 뽑을 때 이
        경로를 그대로 재사용하도록 통일함(이전엔 forward를 별도로 통째로 다시
        구현해서 썼는데, 같은 로직이 두 곳에 중복돼 있었음 - 이제 여기 하나로 합침).
        """
        x = self.conv1(x)             # Input Layer - Figure2 박스에 +ReLU 라벨 없음(활성화 미적용)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2b(x))
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

        x = self.global_max_pool(x)
        x = self.flatten(x)

        fc1_feat = self.fc1(x)
        fc2_feat = self.fc2(x)
        fv3 = torch.cat([fc1_feat, fc2_feat], dim=1)

        logits = self.classifier(fv3)
        if return_features:
            return logits, {"fc1": fc1_feat, "fc2": fc2_feat, "fv3": fv3}
        return logits


def _expected_shapes(input_size=56, batch_size=2, num_classes=3, out_channels=64, num_pools=1):
    """torch 없이도 shape 계산식만 검증할 수 있도록 분리한 순수 함수 (셀프 체크용)."""
    pooled = input_size // (2 ** num_pools)
    flatten_dim = out_channels * pooled ** 3
    return {
        "input": (batch_size, 1, input_size, input_size, input_size),
        "after_convs": (batch_size, out_channels, input_size, input_size, input_size),
        "after_pool": (batch_size, out_channels, pooled, pooled, pooled),
        "flatten_dim": flatten_dim,
        "output": (batch_size, num_classes),
    }


if __name__ == "__main__":
    # torch가 설치된 환경(사용자 로컬 GPU 머신)에서 실행 시 실제 forward pass shape 검증
    base_shapes = _expected_shapes(out_channels=64, num_pools=1)
    variant2_shapes = _expected_shapes(out_channels=512, num_pools=2)
    print("[계산값] Base 예상 shape:", base_shapes)
    print("[계산값] Variant1 예상 flatten_dim: 128 (Global max pooling - 채널 수만 남음, 공간 정보 소거)")
    print("[계산값] Variant2 예상 shape:", variant2_shapes)
    try:
        dummy = torch.randn(2, 1, 56, 56, 56)

        model = CNN3D_Base(num_classes=3)
        out = model(dummy)
        print(f"[실행 검증] Base input={tuple(dummy.shape)} -> output={tuple(out.shape)}")
        assert out.shape == (2, 3), "Base 출력 shape이 예상과 다름"
        print("[통과] CNN3D_Base forward pass 정상 동작.")

        model1 = CNN3D_Variant1(num_classes=3)
        out1 = model1(dummy)
        print(f"[실행 검증] Variant1 input={tuple(dummy.shape)} -> output={tuple(out1.shape)}")
        assert out1.shape == (2, 3), "Variant1 출력 shape이 예상과 다름"
        print("[통과] CNN3D_Variant1 forward pass 정상 동작.")

        model2 = CNN3D_Variant2(num_classes=3)
        out2 = model2(dummy)
        print(f"[실행 검증] Variant2 input={tuple(dummy.shape)} -> output={tuple(out2.shape)}")
        assert out2.shape == (2, 3), "Variant2 출력 shape이 예상과 다름"
        print("[통과] CNN3D_Variant2 forward pass 정상 동작.")

        model3 = CNN3D_Variant3(num_classes=3)
        out3 = model3(dummy)
        print(f"[실행 검증] Variant3 input={tuple(dummy.shape)} -> output={tuple(out3.shape)}")
        assert out3.shape == (2, 3), "Variant3 출력 shape이 예상과 다름"
        print("[통과] CNN3D_Variant3 forward pass 정상 동작.")
    except NameError:
        print("[안내] torch가 설치되지 않은 환경입니다. 계산값만 확인했습니다.")
