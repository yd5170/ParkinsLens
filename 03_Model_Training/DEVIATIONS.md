# 모델링 코드 재작성 - 논문과의 차이점 및 결정 사항 기록

작성일: 2026-07-03
대상: 03_Model/models.py, dataset.py, classifiers.py, train.py, smoke_test.py,
      02_FeatureEngineering/fusion.py, feature_optimization.py

프로젝트 지침 6항("새로운 전처리나 임의의 변경은 적용하지 말고, 필요한 경우 반드시
이유를 명시한 후 진행")에 따라, 논문에 명시되지 않아 자체적으로 결정한 모든 사항을
이 문서에 기록합니다. 논문에 명시된 값은 모두 05_Document/모델_아키텍처_분석.md를
그대로 따랐습니다.

## 1. 3D-CNN (models.py: CNN3D)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 필터 진행(32-64-128-256-512-1024-512-256) | 명시(Figure 2) | 그대로 사용 | - |
| 커널 크기 (3,3,3) | 명시(Figure 2) | 그대로 사용 | - |
| MaxPool (2,2,2) x3 | 명시(Figure 2) | 그대로 사용 | - |
| Stride, Padding | **미명시** | stride=1, padding=1(SAME) | Figure 2의 단계별 출력 크기가 pool 전까지 불변으로 표기되어 있어, 이를 만족하는 유일한 표준 조합으로 채택 |
| Flatten 차원(12544) | 명시(Figure 2)하나 산술 불일치(7×7×7×256=87808) | AdaptiveAvgPool3d로 depth축만 1로 축소 → 7×7×256=12544 | 논문 명시 수치(12544)를 재현 목표로 우선시. flatten_mode="full"(87808) 옵션도 병행 제공 |
| FC-1, FC-2 = 1000 features | 명시(본문 p.5) | 그대로 사용 | - |
| FV-3(FC-1,FC-2 결합) 차원 | Eq.2는 elementwise max(→1000)이나 Fig.1 라벨은 ×2000 | concatenation 채택(2000차원) | 두 서술이 불일치하며, 최종 차원이 2000이 되는 concatenation을 채택(모델_아키텍처_분석.md 불일치사항 3 참조) |

## 2. 3D-ResNet (models.py: ResNet3D)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 15 layers = 5 Residual Block × 3 unit | 명시(본문) | 그대로 사용 | - |
| 채널 수(필터 폭) | **전혀 미명시**(Figure 3에 수치 없음) | 64→128→256→512→512 (표준 ResNet 더블링) | 논문에 근거가 없어 통상적인 ResNet 채널 확장 관행을 자체 적용 |
| 커널 크기 | 미명시 | (3,3,3), 3D-CNN과 동일 통일 | 일관성을 위한 자체 결정 |
| Residual Unit 내부 구조(main path 2conv + skip path 1conv) | 명시(Figure 3 확대도) | 그대로 구현 | - |
| FC-4 차원 | 본문(1000) vs Fig.1(×2000) 불일치 | 1000 채택 | 본문 서술("1000 features are extracted from the 14th layer")이 더 구체적 근거이므로 채택 |
| Block 간 MaxPool | 명시(구조상 존재) | 채택, 마지막은 AdaptiveAvgPool3d(1,1,1) | 최종 크기를 고정하기 위한 자체 결정(입력 크기 변화에 견고) |

## 3. CCA 특징 융합 (fusion.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| 공분산행렬 기반 CCA 절차 | 명시(Eq.3-7) | scikit-learn CCA(NIPALS 알고리즘) 사용 | 수학적으로 동일한 절차, 검증된 구현체 사용 |
| CCA 성분 개수(n_components) | **미명시** | min(n_samples-1, 100) | 학습 샘플 수 및 두 특징 차원(2000, 1000)에 안전하게 맞춘 자체 결정 |
| Z1, Z2 결합 방식 | 미명시 | concatenate | 논문이 결합 방식을 서술하지 않아 가장 단순한 방식 채택 |

## 4. WOA 특징 최적화 (feature_optimization.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| b, threshold, Population, Iteration, bounds | 명시(Table 7) | 그대로 사용(100, 200, [0,1], thres=0.5) | - |
| Fitness 가중치 alpha | **미명시** | 0.99 | WOA 특징선택 문헌의 통상값(정확도에 절대적 가중치) |
| Fitness 내부 KNN의 K값 | **미명시** | 5 | scikit-learn 근사 기본값 |
| 독립 실행 횟수(20회) | 명시(Table 7) | train.py에서 반복 실행 시 적용 예정(smoke test는 1회) | 전체 실행은 GPU 환경에서 수행 |

## 5. 분류기 (classifiers.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| SVM/KNN/GB/RF 하이퍼파라미터 | **전혀 미명시** | scikit-learn 기본값 | 논문에 근거 없음 |
| 평가지표 평균 방식 | 미명시 | weighted average | 클래스 불균형(110/58/135) 고려, 기존 프로젝트 세션 관행 유지 |

## 6. 학습 설정 (train.py)

| 항목 | 논문 명시 여부 | 채택 값 | 근거/이유 |
|---|---|---|---|
| CNN LR=0.001, ResNet LR=0.0001, Epoch=30, L2=0.0001 | 명시(Table 5) | 그대로 사용 | - |
| Optimizer=Grid search 자체의 재실행 | 명시(Table 5)이나 재탐색은 미수행 | 논문이 보고한 최종 채택 설정(Adam, 상기 LR)을 직접 사용 | 그리드서치 전체 재실행은 계산비용이 지나치게 크며, 논문이 이미 "최적 설정"으로 보고한 값을 직접 재현하는 것이 프로젝트 목표(재현)에 부합 |
| Batch size | **명시됨** (Table 3, Study 2) | Base/Variant1/Variant2=64, **Variant3=32** | *(2026-07-22 재정정)* 이전(2026-07-03) 기록은 "Variant3(3D-CNN 최고 성능 구성, Accuracy 93.41%)가 Batch=64"라고 했었는데, 이번 세션에서 Table3 Study2 표를 400dpi 고해상도로 렌더링해 4개 열(Base/V1/V2/V3) 체크 표시를 직접 재대조한 결과 **Variant3만 32이고 나머지 셋이 64**로 확인됨(반대로 기억하고 있었음 - pdftotext 오독이 원인으로 추정). 물리 배치를 그대로 못 올릴 만큼 크다는 이전 우려(24GB+)는 Variant3가 실제로는 더 작은 배치(32)를 쓴다는 게 확인되면서 근거가 약해짐 - gradient accumulation 없이 물리 배치 그대로 사용 중(03_Model_Training/train_ablation.py, train_resnet.py 참조). |
| k-fold(CNN/ResNet 학습 자체) | 명시(k=5) | **2026-07-23 구현 완료.** train_ablation.py/train_resnet.py에 `--kfold 5`(또는 10/15) 옵션 추가 - dataset.py의 신규 `get_kfold_splits_with_val()`로 fold마다 (train, val, test) 3분할을 만들고, fold마다 독립적으로 학습(동일 하이퍼파라미터) 후 fold별 결과 + 평균/표준편차를 집계 JSON(`results/kfold_{variant}_k{K}_*.json`)으로 저장 | *(2026-07-23 이전 기록 정정)* 과거엔 "Table5의 k=5는 그리드서치 내부 검증용으로 해석, 재구현 미수행"이라고 판단했었으나, 사용자가 재확인을 요청해 실제로 구현함. **자체 결정 사항**: Table5는 "k-fold CV=5"만 명시하고 학습 자체의 조기종료/체크포인트 선택에 쓸 validation을 fold 구조 어디서 떼어내는지는 기재하지 않음 - Model-3/4용 `get_kfold_splits()`와 동일한 fold 분할을 쓰고, 남은 train 쪽에서 val_frac(15%)만큼만 추가로 떼어내는 nested holdout 방식 채택(데이터가 이미 적어 별도 validation fold를 또 만들면 표본이 너무 작아짐). 기본 실행(옵션 미지정)은 여전히 단일 holdout - 비교 기준선 유지 목적으로 병행. |
| Model 3/4 평가 k-fold(5/10/15) | 명시(Table 6) | 그대로 구현(dataset.py get_kfold_splits) | - |
| Epoch별 precision/recall/f1(macro) 기록 | 논문 미기재(학습 로그 형식은 논문 범위 밖) | **2026-07-23 추가.** train_loop()에서 매 epoch마다 train/val 양쪽의 macro precision/recall/f1을 계산해 history에 기록(이전엔 loss/acc만 기록) | 친구 코드베이스의 `training_history_ResNet.csv`가 이 지표들을 이미 epoch 단위로 기록하고 있어 비교가 필요했음 - 추가 forward pass 없이 run_epoch()이 이미 반환하는 preds/labels로 계산(비용 없음) |

## 7. 스모크 테스트 관련 (CPU 환경 제약)

- 본 프로젝트의 코드 검증 환경은 GPU가 없는 2-core CPU 환경으로, 실제 30-epoch 전체
  학습은 수행하지 않았습니다(사용자 요청에 따름).
- 실제 마운트된 06_resized 데이터 폴더("전처리06_리사이즈_최종_v2")에서 파일 콘텐츠
  직접 읽기가 간헐적으로 실패하는 환경 이슈(한글 경로명 관련, 메타데이터 조회는
  성공하나 파일 열기가 실패)가 있어, 스모크 테스트는 실제 CSV의 sample_id/label
  메타데이터는 그대로 사용하되, 실제 MRI 볼륨 픽셀 값은 동일 shape(56,56,56)의
  synthetic(무작위) 데이터로 대체하여 코드 동작만 검증했습니다. 이는 코드의 기계적
  정합성(shape 흐름, backward, optimizer step)만을 검증하는 것이 목적이며, 실제 학습
  정확도는 검증 대상이 아닙니다.
- CNN 배치=2에서 Optimizer 1 step에 약 21초가 소요되었고, ResNet은 배치=2에서 메모리
  부족(OOM)으로 실패하여 배치=1로 축소 후 검증했습니다. 이는 CPU 전용/비최적화 빌드
  (Ubuntu 22.04 apt 패키지의 PyTorch 1.8.1, MKL 미포함)의 한계이며, RTX 4060 GPU
  환경에서는 문제가 되지 않을 것으로 예상됩니다.

## 8. Base CNN 학습 불안정성(클래스 붕괴) 대응

2026-07-19, `train_ablation.py`의 Base 모델 실행에서 시드에 따라 특정 클래스
(Control 또는 Prodromal)의 recall이 0으로 완전히 붕괴하는 현상을 발견하고
`calibrate_classifier_scale()`(classifier 출력 스케일 사후 보정, 모델 구조·
하이퍼파라미터는 불변)로 완화함. 원인 분석, 수정 내용, 보정 전/후 비교표,
성능이 개선된 이유(메커니즘)는 별도 문서
[`클래스_붕괴_분석_및_대응.md`](클래스_붕괴_분석_및_대응.md)에 상세 기록.

## 9. Ablation Study 모델(ablation_models.py: Base/Variant1/Variant2/Variant3) 채널 폭

Table3 Study1은 각 variant의 conv/pooling **개수**만 명시하고("two 3D-conv layers" 등),
채널 폭(필터 수)은 전혀 명시하지 않음. Figure2는 최종 완성 모델(8 conv, models.py의
CNN3D)의 채널 진행(32→64→128→256→512→1024→512→256)만 그려져 있고, Base처럼 conv가
2개뿐인 축소 단계에 대한 그림은 논문에 없음 - 이 자체가 논문 전체의 공백 구간.

**채택 원칙**: Base부터 시작해 conv를 하나씩 추가해 나가는 ablation 절차의 성격상,
"하나의 채널 시퀀스가 계속 이어지고 각 variant는 그 앞부분에 해당한다"는 원칙을
채택(2026-07-22, 사용자 확인) - 32→64→128→256→512로 이어지는 단일 시퀀스를 상정하고:

| 모델 | conv 채널 | 원칙 준수 여부 |
|---|---|---|
| Base(conv 2개) | 32, 64 | 준수 |
| Variant1(conv 3개) | 32, 64, 128 | 준수 |
| Variant2(conv 2+3개) | 32,64 \| 128,256,512 | 준수 |
| Variant3(conv 2+3+2개) | 32,64 \| 128,256,512 \| **512,256,128** | **이탈** |

**Variant3의 마지막 블록(conv 2개)이 원칙을 어김** - 512→1024→2048로 계속 늘어나야
하는데, Variant2에서 이미 채널 512까지 커지면서 불안정성을 겪은 것을 우려해
512→256→128로 자체적으로 줄임(2026-07-21). 이는 "Base부터 이어붙인다"는 위 원칙과
어긋나는 임의 이탈이며, 아직 미해결 - 원칙대로 되돌릴지(512→1024→2048, 파라미터
급증·불안정성 재현 우려) 현재 구조(512→256→128, 안정성 우선이나 원칙 이탈)를 유지할지
결정 필요. 되돌릴 경우 Variant3 재학습 필요.

**[2026-07-23 추가] models.py의 CNN3D(최종모델)와 이 ablation_models.py의
CNN3D_Variant3는 "24-layer 모델"이라는 같은 대상을 서로 다른 해석으로 재구현한
별개의 파일임 - 혼동하기 쉬워 명시적으로 정리(사용자 제공 정리표,
`C:\Users\user\Pictures\figure2add.png` 기준):

| 항목 | Figure2 실제(models.py CNN3D) | ablation_models.py CNN3D_Variant3 |
|---|---|---|
| Conv 채널 진행 | 32-64-128-256-512-1024-512-256 | 32-64 / 128-256-512 / 512-256-128 |
| Flatten 방식 | 직접 Flatten = 12544 (depth_collapse) | Global Max Pooling -> 128 |
| FC-1/FC-2 입력 차원 | 12544 | 128 |

`models.py`의 CNN3D는 Figure2 다이어그램의 채널 수·Flatten 차원(12544)을 그대로
재현하는 쪽(위 1번 섹션), `CNN3D_Variant3`는 Table3 Study2가 명시한 그리드서치
결과(Flatten=Global max, 채널 최대 512)를 재현하는 쪽(이 섹션)을 택함 - 각자 근거가
다른 별개의 해석이며 어느 한쪽이 틀린 게 아님. 두 파일의 체크포인트/구조를 섞어 쓰지
말 것(shape mismatch 발생). `models.py`의 `CNN3D` 클래스 docstring에도 동일 내용을
추가해둠.
