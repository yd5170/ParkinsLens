# Variant1 하이퍼파라미터 교정 + 조기종료 버그 수정 + 전처리 육안검증 — 세션 종합 기록

작성일: 2026-07-20
관련 문서: [`DEVIATIONS.md`](DEVIATIONS.md), [`클래스_붕괴_분석_및_대응.md`](클래스_붕괴_분석_및_대응.md),
[`용어설명_시드와_클래스붕괴.md`](용어설명_시드와_클래스붕괴.md),
[`세션_기록_전처리_재구현_및_CNN_재학습.md`](세션_기록_전처리_재구현_및_CNN_재학습.md)

---

## 1. Variant1 하이퍼파라미터 오류 발견 및 교정

### 1-1. 무엇이 잘못돼 있었나

이전 세션까지 `CNN3D_Variant1`의 Table 3 Study 2 하이퍼파라미터를 **Base 모델과 동일값으로
잠정 가정**해 두고 있었다(Pooling=Max, Batch=32, Flatten=Flatten, LR=0.01). `pdftotext`로
논문 표를 추출하면 체크박스 표시가 깨져서 자동으로 읽어낼 수 없었기 때문.

사용자가 논문 PDF의 Table 3 Study 2 표에서 Variant1 열의 체크 표시를 **직접 육안으로 확인**한
결과, 실제 값은 전부 달랐다:

| 항목 | 기존 가정(잘못됨) | 실제(육안 확인) |
|---|---|---|
| Pooling | Max | **Average** |
| Flatten | Flatten | **Global max** |
| Batch size | 32 | **64** |
| Learning rate | 0.01 | **0.001** |
| Activation | ReLU | ReLU (동일) |
| Optimizer | Adam | Adam (동일) |

Flatten이 "Global max"라는 것은 `AdaptiveMaxPool3d(1)`로 공간 정보를 완전히 소거하고
채널 수(128)만 남긴다는 뜻이라, classifier 입력 차원이 **2,809,856 → 128**로 줄어드는
구조적으로 큰 차이였다.

### 1-2. 코드 수정 내역

**`02_Model_Definition/ablation_models.py` — `CNN3D_Variant1`**
- `nn.MaxPool3d` → `nn.AvgPool3d(kernel_size=2, stride=2)`
- `nn.Flatten()` 직전에 `nn.AdaptiveMaxPool3d(1)` 추가(Global Max Pooling)
- `classifier`를 `Linear(2_809_856, num_classes)` → `Linear(128, num_classes)`로 변경
- 파라미터 수: 약 8.4M → **278,211개**로 감소(스모크 테스트로 확인)

**`03_Model_Training/train_ablation.py`**
- `paper_study2["variant1"]`을 위 표의 실제값으로 정정(accuracy/precision/recall/f1은
  기존에 이미 맞게 등록돼 있었음: 88.25/89.93/84.28/87.43)
- `variant_arch_desc`에 `pool_type`/`flatten_type` 키 추가, 비교표의 "Pooling 종류"/
  "Flatten 방식" 행이 더 이상 Base 값을 하드코딩하지 않고 variant별로 동적으로 표시되도록 수정
  (기존엔 이 두 행이 항상 "Max (MaxPool3d)"/"Flatten"로 고정 출력되던 버그)
- `VARIANT_DEFAULT_HP` 딕셔너리 신설 — `--variant` 선택에 따라 `--batch_size`/`--lr`
  기본값이 자동으로 맞는 조합(Base=32/0.01, Variant1=64/0.001)을 쓰도록 변경. 이전엔
  두 옵션의 기본값이 32/0.01로 고정돼 있어 `--variant variant1`을 지정해도 직접
  `--batch_size 64 --lr 0.001`을 안 붙이면 Base 조합으로 도는 문제가 있었음

수정 후 1-epoch 스모크 테스트로 배선을 검증: 비교표의 Pooling/Flatten/Batch/LR 4개 항목
전부 "일치"로 뜨는 것을 확인.

---

## 2. Early stopping 도입 — 첫 구현의 버그와 진단

### 2-1. 요청과 최초 구현

사용자 요청: "과적합이 5번이상 일어나면 에폭을 더 늘리지 않고 멈춰도 되고 베스트 값을 써줘."

최초 구현: `--patience 5`, **val_acc**가 5epoch 연속 개선 안 되면 조기종료, best-checkpoint는
이미 있던 로직(best-validation epoch 가중치로 test 평가) 그대로 사용.

### 2-2. 문제 발생

Variant1 seed42/43을 재실행하니 두 시드 모두 **best epoch이 1~2번째**로, 겨우 6~7epoch만에
조기종료됐고 test accuracy가 각각 36.96%/17.39%로 극히 낮게 나왔다.

사용자 지적: "학습이 거의 이루어지지 않았는데 과적합이라니 무슨말이 되니?" — 정확한 지적이었다.

### 2-3. 원인 진단 (결과 JSON의 epoch별 history 재검토)

`results/ablation_variant1_20260720_195111_acc37.0.json` 등을 다시 열어보니:

- **val_acc가 문자 그대로 고정**돼 있었다. val 샘플이 45개뿐이라 정확도는 1/45(≈2.2%p)
  단위로만 움직이는 양자화된 지표인데, seed42는 6epoch 중 5epoch이 정확히 0.3556(16/45),
  seed43은 7epoch 중 5epoch이 정확히 0.2000(9/45)으로 동일했다.
- 반면 **val_loss는 계속 움직이고 있었다**(seed42: 6.84→8.59→7.63→5.38→4.24→2.59로 꾸준히
  감소). 즉 모델은 실제로 학습되고 있었는데, val_acc라는 지표 자체가 45개 샘플에서는
  변화를 감지하기엔 너무 거칠어서 "5epoch 연속 무변화"로 잘못 판정된 것.
- test 결과의 classification report를 보면 seed42는 Control만 recall 1.00(전부 Control로
  찍음), seed43은 Prodromal만 recall 1.00 — **단일 클래스로의 예측 붕괴(class collapse)**였지,
  "train은 과하게 맞고 val은 나빠지는" 전형적 과적합 패턴이 아니었다.
- 추가로, batch_size=64에 train=212개면 **epoch당 배치가 4개뿐**이라(마지막 배치는 20개),
  patience=5는 사실상 "약 20번의 optimizer 업데이트 동안 무변화면 중단"인 셈이라
  278K 파라미터 모델이 수렴을 판단하기엔 너무 이른 기준이었다. lr=0.001(Base의 0.01보다
  10배 작음)도 수렴을 더 느리게 만드는 요인.
- `calibrate_classifier_scale`은 학습 시작 전 **1회성** 초기화 보정(classifier 출력
  표준편차를 목표값으로 맞추는 것)일 뿐, 학습 중 재발하는 클래스 붕괴를 막는 장치가
  아니라는 것도 확인.

**결론**: "과적합"이라는 조기종료 메시지 문구 자체가 부정확했다. 실제로는 (1) 45개짜리
val셋에서 val_acc가 지나치게 거친 지표였고, (2) batch=64/lr=0.001 조합에서 epoch당
학습 신호가 원래 적어서, 조기종료 설계 자체가 성급하게 판단한 것이었다.

---

## 3. 수정: val_loss 기준 + min_epochs 안전장치

### 3-1. 코드 변경 (`train_ablation.py`)

- best-checkpoint/조기종료 판단 기준을 **val_acc → val_loss**로 변경(연속값이라 45개
  샘플에서도 val_acc보다 훨씬 덜 튐)
- `--min_epochs`(기본 15) 인자 추가 — 이 epoch 수를 채우기 전에는 조기종료를 아예
  허용하지 않는 안전장치. batch가 커서 epoch당 업데이트가 적은 조합에서도 최소한의
  학습 기회를 보장
- 조기종료 로그 문구에서 "과적합 신호"라는 인과 단정 표현 제거, "validation loss가
  N epoch 연속 개선되지 않아 중단"으로 중립적으로 변경
- 로그(txt/csv)·결과 JSON에 실제로 돈 epoch 수(`actual_epochs`)와 조기종료 여부
  (`early_stopped`)를 남기도록 수정(기존엔 요청한 epoch 수만 기록돼 혼동 가능성 있었음)

### 3-2. 재실행 결과 (Variant1, 교정된 구조 + val_loss 기준 조기종료)

| 항목 | seed42 | seed43 | 평균 | 논문값(Study2) |
|---|---|---|---|---|
| Best epoch (조기종료 시점) | 13 (18에서 중단) | 14 (19에서 중단) | - | 30 |
| Best val_loss | 1.3467 | 1.1816 | - | - |
| Best val_acc | 40.00% | 42.22% | 41.11% | - |
| Test Accuracy | 45.65% | 52.17% | **48.91%** | 88.25% |
| Test Precision (macro) | 50.13% | 47.07% | 48.60% | 89.93% |
| Test Recall (macro) | 49.63% | 47.74% | 48.69% | 84.28% |
| Test F1 (macro) | 40.43% | 47.18% | 43.81% | 87.43% |
| 총 소요시간 | 3805.6s (63.4분) | 3970.8s (66.2분) | - | - |

이번엔 train_acc가 70~80%대까지 정상적으로 상승했고, test 클래스별 recall도 더 이상
한 클래스에 몰리지 않음(seed43: Control 0.71 / Prodromal 0.25 / PD 0.48 — PD만 여전히 약함).

참고로 잘못된 하이퍼파라미터(batch=32, lr=0.01)로 강제로 30epoch 다 돌렸던 이전 결과는
52.17%/41.30%(평균 46.74%)였다 — 교정 후 평균(48.91%)과 비슷한 수준이지만, 이번엔
클래스 붕괴 없이 정상 수렴한 뒤의 값이라 방법론적으로 더 신뢰할 수 있다.

### 3-3. (참고) Variant2 seed42 30epoch 결과

Variant1 작업과 별도로 이미 완료돼 있던 Variant2(하이퍼파라미터는 아직 육안 미확인,
Base와 동일값 32/0.01로 잠정 가정) seed42 결과:

- Best epoch 18/30, val_acc 51.11%, Test Accuracy 47.83%, F1(macro) 50.42%
- 논문값(Study2, Variant2): Accuracy 90.28%, F1 90.81%

---

## 4. 논문값과의 격차 — 원인 후보 분석

Base(~50%대) / Variant1(평균 48.91%) / Variant2(단일실행 47.83%) 모두 논문 대비 큰 격차가
있고, 특히 **레이어가 늘어날수록 논문처럼 좋아지는 경향이 우리 재현에서는 안 보인다**
(논문: Base 82.02% < V1 85.75% < V2 88.76%, 우리: 셋 다 비슷하거나 오히려 하락).
구조 자체는 이번에 육안 재검증했으니 구현 오류 가능성은 낮아, 네 모델이 공통으로 겪는
입력 데이터/파이프라인 쪽 병목일 가능성이 높다고 판단. 후보:

1. **전처리 재구성의 한계**(가장 유력, 4절 육안검증에서 구체적 근거 발견) — 논문 자체의
   전처리 서술이 Methods/Results가 서로 다르게 기술돼 있어(`DEVIATIONS.md` 참조),
   우리 `전처리_ref21order_v1`은 참고문헌 기반 최선의 재구성이지 논문의 실제 파이프라인과
   동일하다는 보장이 없음
2. **데이터 증강(augmentation) 부재** — 논문 Methods에 "data augmentation" 언급이 있으나
   구체적 방법 미기재라 우리는 구현하지 않음. 303명 규모에서는 영향이 클 수 있음
3. **분할(split) 전략 차이** — 논문이 k-fold 교차검증 평균을 Study2 값으로 보고했을
   가능성 배제 못함. 우리는 단일 holdout(212/45/46)이라 원래 더 낮고 불안정한 값이 나옴
4. **평가셋이 작아 지표 자체가 거침** — val 45개/test 46개는 샘플 하나가 ~2.2%p라
   지표가 원래 노이즈가 큼(2-3절 참조)
5. **논문 Study2 값 자체가 그리드서치 중 최적 조합**일 가능성 — 우리는 시드 2개 평균이라
   비교 기준이 다름

---

## 5. 전처리 파이프라인 육안 검증

### 5-1. 방법

Control(`sub-3000_I224561`)/Prodromal(`sub-16785_I814554`)/PD(`sub-3001_I224572`) 각 1명씩
골라 BET → 정합(registration) → min-max 정규화 → 최종(56³ 리사이즈) 4단계를
sagittal/coronal/axial 중앙 슬라이스로 나란히 시각화.

- 스크립트: [`../01_Preprocessing/스크립트/qc_visual_stages.py`](../01_Preprocessing/스크립트/qc_visual_stages.py)
- 결과 이미지: [`../01_Preprocessing/QC_전처리단계_시각화_3샘플.png`](../01_Preprocessing/QC_전처리단계_시각화_3샘플.png)
- 중간단계 파일이 `_work/` 하위에 남아있는 샘플 중에서 그룹별 1명씩 선정(중간 파일을
  지운 샘플은 최종본만 남아있어 단계별 비교 불가)

### 5-2. 관찰 결과

- **원본 자체가 slice 수가 매우 적음**: Control 원본이 (512,512,**18**), Prodromal/PD는
  (512,512,**26**). PPMI raw DICOM이 2D 다중슬라이스 T2 촬영이라 z축(슬라이스 방향)
  해상도가 원래 낮음(in-plane 512×512는 고해상도, slice 방향은 18~26장뿐). BET 단계의
  sagittal/coronal 중앙 슬라이스가 얇은 띠처럼 눌려 보이는 게 이 때문
- **BET(두개골 제거)·정합은 정상 작동**하는 것으로 보임 — axial 뷰에서 두개골이 깔끔히
  제거됐고, 정합 후 세 피험자 모두 (193,229,193) 표준(atlas) 공간으로 잘 정렬돼 좌우
  대칭·구조 위치에 큰 오정합은 안 보임
- **의심되는 지점 — 최종 56³ 리사이즈**: (193,229,193)→(56,56,56)이면 복셀 하나가
  약 3.4mm. T2 영상에서 PD 진단의 핵심 신호로 알려진 흑질(substantia nigra)의
  nigrosome-1 sign은 크기가 수 mm 수준이라 이 해상도에서는 사실상 소실될 가능성이 큼.
  게다가 원본이 애초에 slice 방향으로 18~26장뿐이라, 정합 단계에서 (193,229,193)으로
  늘어난 부분 대부분이 보간(interpolation)으로 채워진 것이지 실제 촬영 정보가 아님 —
  z축 방향의 "진짜" 정보량이 처음부터 부족한 상태에서 한 번 더 다운샘플링되는 셈.
  이는 Variant1 test 결과에서 PD 클래스 recall이 유독 약했던 것(seed42: 0.05, seed43: 0.48)과
  정황상 맞아떨어짐
- 다만 56³은 논문 아키텍처(Table3의 레이어별 차원 계산 근거)가 요구하는 입력 크기라,
  이는 우리 파이프라인만의 문제가 아니라 **논문이 원래 설계부터 감수했을 가능성이 있는
  제약**으로 보임. 아키텍처를 벗어나지 않고는 이 부분을 바꾸기 어려워 신중한 접근이 필요.

---

## 6. 남은 작업 (Pending)

- Variant2 하이퍼파라미터(Pooling/Flatten/Batch/LR)도 Variant1과 마찬가지로 아직
  논문 표 육안 확인 전 — Base 값으로 잠정 가정 중. 확인 필요
- Variant2 seed43 미실행
- 이번 세션에서 변경된 코드(`ablation_models.py`, `train_ablation.py`,
  `qc_visual_stages.py` 신규)와 이 문서를 GitHub에 아직 푸시하지 않음
- 4절의 원인 후보 중 어느 것이 실제로 지배적인지는 아직 검증 전(가설 단계)
