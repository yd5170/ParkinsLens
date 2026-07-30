# Base/Variant2 배치사이즈 재정정 + Variant3 구현 + ResNet(Model-2) 착수 — 세션 종합 기록

작성일: 2026-07-22
관련 문서: [`DEVIATIONS.md`](DEVIATIONS.md),
[`세션_기록_Variant1_교정_및_조기종료_버그수정.md`](세션_기록_Variant1_교정_및_조기종료_버그수정.md)

---

## 1. Base/Variant2 배치사이즈 재정정 (32 → 64)

이전 세션에서 "Table3 Study2 Base 열을 육안 확인한 결과 배치=32"라고 기록해뒀던 게 **잘못 읽은
것**이었음이 드러남. 표를 400dpi로 고해상도 렌더링해 Base/Variant1/Variant2/Variant3 4개 열을
동시에 재대조한 결과:

| 항목 | Base | Variant1 | Variant2 | Variant3 |
|---|---|---|---|---|
| Batch size | **64**(이전 32로 오독) | 64 | **64**(이전 32로 잠정가정) | 32 |

Base/Variant3가 서로 반대로 잘못 읽혔던 것 — 실제로는 Base=64, Variant3만 32로 다름.
Variant2도 "Base와 동일값"으로 32를 잠정 가정했던 게 결과적으로 틀렸음(64가 맞음).
`train_ablation.py`의 `VARIANT_DEFAULT_HP`를 정정하고, Base·Variant2를 batch=64로 재실행함.

---

## 2. Variant3(24-layer) 구현

### 2-1. 구조를 둘러싼 혼란과 해소 과정

Ablation 본문의 Variant3 서술("2+3+2 conv, flatten dense layer, SoftMax layer")만으로 계산하면
23층이 나와 논문이 명시한 24층과 1개 차이가 났음. 사용자가 논문의 "Proposed 3D CNN" 다이어그램
(Figure 2와 동일 대상이지만 각 레이어가 개별 박스로 명확히 표기된 버전)을 캡처해 제공.

이 다이어그램으로 확정된 사실:
- conv 블록 구성이 실제로 "2+3+2"(Ablation 본문과 일치)
- "flatten dense layer"라는 표현은 실제로는 Flatten 이후 **FC-1, FC-2 두 개의 병렬
  Dense(각 1000개)를 concat(2000)한 것**을 가리키는 축약 표현이었음
- 층수 재검산: input(1) + [2conv+2활성화+1pool+1BN=6] + [3conv+3활성화+1pool+1BN=8] +
  [2conv+2활성화+1pool+1BN=6] + FC-1(1) + FC-2(1) + SoftMax(1) = **정확히 24** (일치).
  이전의 23은 FC-1/FC-2를 빼먹은 계산 실수였지 논문 불일치가 아니었음.

### 2-2. 구현 (`ablation_models.py`의 `CNN3D_Variant3`)

- 블록1(conv 1→32→64) → AvgPool → BN(64)
- 블록2(conv 64→128→256→512) → AvgPool → BN(512)
- 블록3(conv 512→256→128) → AvgPool → BN(128) — 채널을 다시 줄임(Variant2에서 채널을
  계속 키우기만 해서 생겼던 과대 파라미터 문제의 교훈 반영, 최종모델 `models.py`의
  마지막 블록 감소 패턴 참고)
- Global Max Pooling(Study2 확정값) → FC-1(128→1000) + FC-2(128→1000) 병렬 → concat(2000)
  → classifier(2000→3)
- Study2 확정 하이퍼파라미터(육안 확인): Pooling=Average, Batch=32, LR=0.001, Optimizer=Adam

### 2-3. 결과 (seed42, 30epoch 끝까지, val_loss 기준 best)

| | Best Epoch | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Variant3 | 28/30 | 45.65% | 44.72% | 46.98% | 44.47% |
| 논문 Study2 | 30 | 93.41% | 95.76% | 88.98% | 91.68% |

---

## 3. 4개 모델 최종 비교 (batch/구조 전부 정정 완료, seed42, 30epoch 끝까지, val_loss 기준)

| 모델 | 파라미터 수 | 평균 Epoch 시간 | 총 소요시간 | Best Epoch | Accuracy | Precision | Recall | F1 | 논문 Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Base | 4,271,171 | 32.3s | 968.6s | 29/30 | 52.17% | 61.35% | 56.22% | 54.37% | 84.96% |
| Variant1 | 278,211 | 201.6s | 6049.4s | 23/30 | 56.52% | 57.20% | 60.11% | 57.46% | 88.25% |
| Variant2 | 8,917,955 | 189.5s | 5684.5s | 26/30 | 26.09% | 17.55% | 34.56% | 22.38% | 90.28% |
| Variant3 | 9,391,491 | 41.1s | 1233.6s | 28/30 | 45.65% | 44.72% | 46.98% | 44.47% | 93.41% |

**주목할 점**: 파라미터 수와 epoch당 시간이 비례하지 않음 — Variant1(파라미터 가장 적음,
27.8만개)이 오히려 가장 느리고(201.6s/epoch), Variant3(934만개, Variant1의 34배)가 훨씬
빠름(41.1s/epoch). conv 레이어의 공간해상도×채널 구성이 파라미터 총량보다 속도에 더 큰
영향을 주는 것으로 보이나, 정확한 원인은 별도 조사 필요.

논문은 레이어가 늘수록 꾸준히 좋아지는데(82.02→88.25→90.28→93.41), 우리 재현은 이 경향이
전혀 안 보임(오히려 Variant2가 가장 낮음). 하이퍼파라미터/구조는 이번에 전부 육안 재검증
했으므로, 남은 격차는 전처리 재구성 한계·데이터 증강 부재 등 입력 데이터 쪽 요인일 가능성이
높다고 판단(상세: [`세션_기록_Variant1_교정_및_조기종료_버그수정.md`](세션_기록_Variant1_교정_및_조기종료_버그수정.md) 4절).

---

## 4. cnn_training_report 엑셀 통합

사용자가 별도 코드베이스(`D:\Brain_Tensor\3dcnn_results\paper_like_low_memory`)로 이미 돌려둔
`cnn_training_report.xlsx`(9개 시트: Model Summary, Epoch Metrics, Best Validation, Model
Structure, Paper Architecture, Confusion Matrix, Run Settings, Assumptions, Split Counts)를
확인하고, 우리 세션(`D:\new tensor`) 결과를 같은 양식으로 정리해달라는 요청을 받음.

**원본은 절대 건드리지 않고**, 복사본(`cnn_training_report_통합비교.xlsx`)에 모든 시트마다
`Source` 컬럼을 추가해 우리 세션 행을 구분(`이번 세션(new tensor, 2026-07-21)`)해서 추가함.

- Model Summary/Confusion Matrix: 체크포인트(.pt)를 다시 불러와 test셋 재추론까지 해서
  **weighted precision/recall/f1과 실제 confusion matrix**까지 정확히 채움. 이를 위해
  Base·Variant2도 (이전엔 체크포인트가 없어서) seed42로 재실행해 체크포인트 확보.
- Model Structure/Paper Architecture: 우리 실제 PyTorch 코드를 직접 introspect해서 정확히 채움.
- **정직하게 "없음"으로 남긴 부분**: Epoch Metrics·Best Validation의 precision/recall/f1
  (macro/weighted 전부) — 우리는 epoch마다 loss/accuracy만 기록했고 validation set의
  precision/recall/f1은 계산한 적이 없음. 존재하지 않는 값을 지어내지 않고 전부 "없음"으로 표기.

---

## 5. ResNet3D(Model-2) 구현 착수 — 논문 후속 파이프라인 1단계

### 5-1. 배경

논문은 ablation study(Base~Variant3) 이후 별도의 대형 파이프라인을 거쳐 최종 97.2% 정확도에
도달함:
1. **Improved 3D-ResNet** 구현·학습 (Model-2)
2. 3D-CNN(Variant3)의 FC 특징 + 3D-ResNet의 FC 특징을 **CCA**로 융합
3. 융합 특징으로 **Gradient Boosting** 분류 (Model-3, 최적화 전 ~90%)
4. **WOA**(Whale Optimization Algorithm)로 융합 특징 중 최적 특징 선택 후 재분류 (Model-4, 97.2%)

사용자 결정: 1단계(ResNet 구현)부터 순서대로 진행.

### 5-2. 발견: ResNet3D는 이미 구현되어 있었음

`02_Model_Definition/models.py`에 `ResidualUnit3D`/`ResidualBlock3D`/`ResNet3D` 클래스가
이전 세션에 이미 작성돼 있었음(미등록·미테스트 상태). 논문 원문(Figure3 캡션 주변) 대조:

> "This neural network architecture is composed of 15 layers, including an input layer,
> two conv3D blocks, five residual blocks with max-pooling layers, a fully connected layer,
> and a SoftMax layer. Each 'conv3D block' consists of a 3D conv layer, BN, and ReLU
> activation layers. The residual block contains three residual units, each comprised of
> two 3D Conv layers, BN, and ReLU activation layers... skip connection... adds the input
> to the output of the last ReLU layer."

층수 검산: 1(input)+2(conv3D 블록)+5(residual block)+5(maxpool, 블록마다 1개)+1(FC)+1(softmax)=15,
일치. "14번째 층에서 1000개 특징 추출"은 FC 레이어(14번째)를 가리키는 것으로 해석.

### 5-3. 사용자 제공 Figure3 다이어그램으로 발견/정정한 것 2가지

1. **pool4가 AdaptiveAvgPool3d로 잘못 구현돼 있었음** — 논문은 5개 블록 전부 max-pooling이라고
   명시. `nn.MaxPool3d(kernel=2,stride=2)`로 정정(3→1, 산술상 문제없음 확인).
2. **FC-4 출력 차원이 1000 → 2000으로 정정** — 논문 "Feature extraction" 본문은 "1000 features
   from 14th layer"라고 하지만, 사용자가 제공한 Figure3 다이어그램은 FC-4 출력을 명시적으로
   **"FV-4 x 2000"**이라고 라벨링함. Figure2(CNN)도 FC-1/FC-2 각 1000을 concat해서 자체적으로
   2000이 되는 구조인데, 이 ResNet 다이어그램은 병렬 브랜치 없이 FC-4 하나로 바로 2000을 출력 —
   본문 텍스트와 다이어그램이 불일치하는 세 번째 사례(1000 vs 2000). 다이어그램(더 구체적·명시적
   라벨)을 우선 채택해 `fc4 = nn.Linear(512, 2000)`, `classifier = nn.Linear(2000, num_classes)`로 정정.
   (참고: CNN의 Variant3 FC-1/FC-2는 이미 각 1000으로 올바르게 구현돼 있어 변경 불필요.)

### 5-4. 학습 스크립트 (`train_resnet.py`)

`train_ablation.py`와 동일한 데이터(`전처리_ref21order_v1`)·동일한 안정화 기법(gradient
clipping, classifier 초기화 스케일 보정, val_loss 기준 best-checkpoint)을 재사용. 논문이
ResNet 자체의 batch/lr을 명시하지 않아 자체 결정(batch=32, lr=0.001 — 파라미터가 9500만개로
매우 커서 작은 배치 사용).

### 5-5. 진행 상황

- 파라미터 수: 95,514,051개 (지금까지 만든 모델 중 최대, Variant3의 약 10배)
- 스모크 테스트(1epoch): 750.9초(12.5분) 소요, val_loss가 26,070,096까지 폭주하며 단일 클래스
  붕괴(PD만 예측) 관찰 — Base 초기에 겪었던 것과 같은 패턴, 정상적인 1epoch째 현상으로 판단
- 30epoch epoch1 기준 예상 총 소요시간: **약 6시간 15분** (지금까지 중 가장 오래 걸림)
- 사용자 결정: 조기종료 없이 30epoch 끝까지 진행(다른 4개 모델과 방법론 통일)
- 논문 목표치(Model-2, ResNet 특징을 Gradient Boosting으로 분류) = 90.0% — 단, 이 스크립트는
  ResNet 자체 SoftMax 분류 결과라 GB 결과와 직접 비교 불가(참고용)
- **현재 30epoch 본실행 진행 중** (완료 후 결과 추가 예정)

---

## 6. 남은 작업 (Pending)

- ResNet3D 30epoch 본실행 결과 대기
- CCA 특징융합, WOA 특징선택, Gradient Boosting 분류(Model-3/Model-4) 전부 미착수
- Variant1/Variant2/Variant3의 seed43 결과 없음(seed42만 "공식" 실행으로 확정)
- 이번 세션 코드 변경사항(Variant3 구현, ResNet 정정, train_resnet.py 신설, 진행상황 print
  추가, stdout line buffering 수정) GitHub 미푸시
