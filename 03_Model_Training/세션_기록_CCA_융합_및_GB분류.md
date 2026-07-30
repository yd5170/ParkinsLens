# ResNet 재학습(lr 정정) + CCA 특징융합 + Gradient Boosting 분류 — 세션 종합 기록

작성일: 2026-07-22
관련 문서: [`세션_기록_Variant3_구현_및_ResNet_착수.md`](세션_기록_Variant3_구현_및_ResNet_착수.md),
[`DEVIATIONS.md`](DEVIATIONS.md), [`오류_수정_히스토리.md`](오류_수정_히스토리.md)

---

## 1. ResNet 재학습 — lr 정정(0.001 → 0.0001)

### 1-1. 무엇이 왜 틀렸었나

ResNet 첫 30epoch 본실행(2026-07-22, 약 5.5시간)을 lr=0.001로 돌렸음. 이 값은 논문에
ResNet 전용 lr이 없다고 보고 CNN(Variant1/3)과 같은 값을 자체결정으로 가져다 쓴 것.

**실제로는 논문 Table 5**("Comparison of training and validation details for 3D CNN and
3D ResNet Models")에 3D CNN과 3D ResNet의 학습 설정이 별도로 명시돼 있었음:

| 항목 | 3D CNN | 3D ResNet |
|---|---|---|
| Learning rate | 0.001 | **0.0001** |
| Epochs | 30 | 30 |
| k-fold CV | 5 | 5 |
| L2(weight decay) | 0.0001 | 0.0001 |

게다가 **이 값은 이미 2026-07-03에 작성된 `DEVIATIONS.md`에 정확히 기록돼 있었음**
("CNN LR=0.001, ResNet LR=0.0001, Epoch=30, L2=0.0001 | 명시(Table 5) | 그대로 사용").
확인 없이 재작업해서 5.5시간짜리 학습을 통째로 버림(상세: `오류_수정_히스토리.md`).

### 1-2. 재학습 결과 비교

| | lr=0.001(폐기) | lr=0.0001(정정, 채택) |
|---|---|---|
| Best epoch | 12/30 | 25/30 |
| Best val_loss | 1.65(그런데 그 시점 val_acc는 최악) | 0.91 |
| Test Accuracy | 15.22% | **54.35%** |
| Precision(macro) | 5.83% | 37.04% |
| Recall(macro) | 29.17% | 45.28% |
| F1(macro) | 9.72% | 39.82% |

train_loss가 수백만~수천만대로 폭주하던 것(lr=0.001)이 lr=0.0001에서는 1~6 범위로
안정됨. lr 하나 바꾼 것만으로 정확도가 15%→54%로 크게 개선 — ResNet이 깊어서(15-layer,
9550만 파라미터) CNN보다 훨씬 작은 lr이 필요했다는 논문 Table5의 판단이 실증적으로도
맞았음을 확인.

체크포인트: `checkpoints/resnet3d_20260722_170643_acc54.3.pt`

---

## 2. FC-3(CNN 융합용 특징)와 FC-4(ResNet 특징) 차원 논쟁 — 최종 해결

### 2-1. 세 번 번복된 판단

이 세션에서 CNN/ResNet의 "융합용 특징 차원"이 몇 번이나 뒤집혔음(`오류_수정_히스토리.md`
참조). 최종적으로 아래 근거들을 교차검증해 확정:

- **Figure2 다이어그램**: "FC-1(1000)+FC-2(1000) → FV-3×2000" 라벨
- **Figure3 다이어그램**: "FC-4 → FV-4×2000" 라벨
- **"Feature extraction" 본문**: "CNN에서 1000개... ResNet에서 1000개... **양쪽 합쳐서
  총 2000개**가 최종 추출"
- **"Feature concatenation" 본문 + Algorithm 1**: FC-3(=CNN 융합 특징)는 Algorithm1로
  만들어지며 "dimensions FV1×1000 and FV2×1000"라고 명시(즉 FC-3 자체가 1000차원)
- **CCA 절**: "3D-ResNet과 3D-CNN 양쪽에서 총 2000개 특징이 추출됐다"(=CCA 입력 직전)
- **Table 4 Analysis 3**: "PD prediction using feature fusion from FC-1 and FC-2 layer
  (**i.e. FC-3**)" — FC-3라는 이름이 명시적으로 이 문서에 정의됨

### 2-2. 결론: 두 개의 서로 다른 파이프라인이 공존

| | Figure2가 보여주는 것 | Table4/Algorithm1이 설명하는 것 |
|---|---|---|
| 목적 | **Variant3 자체의** 3클래스 분류(SoftMax) | **CCA 융합용** 특징(FC-3) 생성 |
| FC-1+FC-2 결합 | 그냥 이어붙임(concat) | Algorithm1(원소별 가중치 비교 선택) |
| 결과 차원 | 2000 | **1000** |
| 다음 단계 | SoftMax → 3클래스(끝) | ResNet의 FC-4(1000)와 CCA 융합 → WOA → 별도 분류기 |

**Variant3 모델 자체(SoftMax 분류용)는 수정 불필요** — 이미 Figure2대로 concat(2000)
구현돼 있었음, 재학습 없이 기존 체크포인트(`ablation_variant3_20260721_225937_acc45.7.pt`)
그대로 사용. **CCA 융합용으로만** Algorithm1을 별도 구현해 FC-1/FC-2에서 1000차원 FC-3를
새로 뽑음. ResNet의 FC-4도 1000으로 최종 확정(`models.py`).

---

## 3. Algorithm 1 구현 (Feature Concatenation)

논문 의사코드를 그대로 구현(`extract_features.py`의 `algorithm1_feature_concatenation`):

```
for j in 0..length(x1)-1:
    weighted_x1 = w1 * x1[j]
    weighted_x2 = w2 * x2[j]
    if weighted_x1 > weighted_x2 and x1[j] not in used_features:
        x3[j] = x1[j]
    elif weighted_x2 >= weighted_x1 and x2[j] not in used_features:
        x3[j] = x2[j]
    else:
        x3[j] = fallback_logic()
```

**논문 미기재라 자체결정한 부분**:
- `w1, w2`(가중치): 동일 가중치 1.0 (논문에 값 없음)
- `Normalize(x1, x2)`: 샘플별 z-score 표준화(평균0, 표준편차1) 채택 — 서로 다른 스케일의
  두 벡터를 비교 가능하게 만드는 가장 표준적인 방법
- `fallback_logic()`: 논문이 이름만 적어두고 정의를 안 함 — 두 후보 다 이미
  `used_features`에 있을 때 가중치 무시하고 `max(x1[j], x2[j])`를 쓰는 것으로 자체결정
  (출력이 항상 정의되도록 하는 가장 단순한 방법)
- `used_features`(중복 방지, "avoiding duplication of elements"): 원문 그대로 **값**
  기준으로 추적(인덱스가 아님) — ReLU 특징은 0.0이 매우 흔해서 두 번째 이후 0.0
  후보는 계속 fallback으로 빠지는 한계가 있으나, 의사코드를 그대로 따름

---

## 4. CCA 특징 융합

- 입력: CNN의 FC-3(1000차원, train 212개) + ResNet의 FC-4(1000차원)
- **n_components=211** — 수학적 최댓값(=n_train-1). CCA는 n개 샘플로 추정한
  공분산행렬의 rank가 n-1을 못 넘기 때문에, 아무리 특징이 1000차원이어도 학습
  샘플이 212개면 독립적인 canonical component를 211개까지만 만들 수 있음
  (사용자 확인 후 채택)
- scikit-learn `CCA(n_components=211)` 사용
- 두 융합 공식(Eq.6, Eq.7)을 논문이 모두 제시하므로 **둘 다 계산**해 비교:
  - Eq(6) Z1 = concat(W_A^T·A, W_B^T·B) → **422차원**
  - Eq(7) Z2 = W_A^T·A + W_B^T·B → **211차원**

---

## 5. Gradient Boosting 분류 — Eq6 vs Eq7 비교 및 분석

### 5-1. 결과

| | Eq(6) concat(422차원) | Eq(7) 합산(211차원) |
|---|---|---|
| Val Accuracy | 31.11% | **46.67%** |
| Test Accuracy | 36.96% | **45.65%** |
| Test Precision(macro) | 34.83% | **47.44%** |
| Test Recall(macro) | 33.30% | **45.18%** |
| Test F1(macro) | 31.74% | **44.67%** |

GB 하이퍼파라미터는 논문 미기재 - scikit-learn `GradientBoostingClassifier` 기본값
사용(`DEVIATIONS.md` 5번 섹션과 동일 근거).

### 5-2. 결과 분석 — 왜 합산(Eq7)이 concat(Eq6)보다 나았나

1. **차원 대 샘플 수 비율**: concat은 422차원인데 학습 샘플이 212개뿐이라 특징이
   샘플보다 2배 많음(과적합 위험 매우 큼). 합산은 211차원으로 딱 샘플 수와 비슷해
   상대적으로 안전함.
2. **CCA의 설계상 합산이 완전히 임의적이지 않음**: CCA는 애초에 "i번째 CNN 성분과
   i번째 ResNet 성분이 최대한 상관되도록" 투영행렬을 학습하므로, 이 둘을 더하는 건
   서로 무관한 값을 더하는 게 아니라 "같은 신호를 두 경로로 측정한 값의 평균"에
   가까움 — 노이즈 상쇄 효과를 기대할 수 있음.
3. concat은 이 상관관계를 활용하지 않고 단순히 정보량만 늘리는 방식이라, 이미 매우
   작은 데이터셋(212개)에서는 정보 보존보다 차원 축소가 더 실익이 컸던 것으로 해석됨.

### 5-3. 전체 파이프라인 비교

| 단계 | Test Accuracy | Test F1(macro) |
|---|---|---|
| Variant3(CNN 단독, SoftMax) | 45.65% | 44.47% |
| ResNet3D(단독, SoftMax) | 54.35% | 39.82% |
| CCA 융합 + GB(Eq6 concat) | 36.96% | 31.74% |
| **CCA 융합 + GB(Eq7 합산)** | 45.65% | 44.67% |

**중요한 발견**: 이번 재현에서는 CCA 융합이 단독 모델보다 뚜렷이 좋아지지 않았음
(Eq7 결과가 Variant3 단독과 accuracy가 정확히 같고 F1만 근소 우위) — 오히려 **ResNet
단독(54.35%)이 지금까지 만든 모든 조합 중 가장 높음**. 논문은 반대로 융합이 단독보다
항상 좋아진다고 보고하는데(Table4: FC-3 단독 93.5% > FC-1/FC-2 각각 87~90%), 이
재현에서는 그 경향이 나타나지 않음. 원인 후보:
- 학습 샘플이 212개로 매우 작아 CCA/GB가 각 모델의 진짜 상관 패턴을 안정적으로
  찾기엔 데이터가 부족했을 가능성
- Variant3/ResNet 각 모델 자체의 성능이 논문 대비 이미 크게 낮아서(45~54% vs 논문
  90%+), 애초에 융합할 만한 양질의 신호가 부족했을 가능성이 큼(전처리 재구성
  한계 등 - `세션_기록_Variant1_교정_및_조기종료_버그수정.md` 4절 참조)

---

## 6. 전체 단계별 소요시간 기록

| 단계 | 스크립트 | 소요시간 | 비고 |
|---|---|---|---|
| ResNet 학습(lr=0.001, 폐기) | `train_resnet.py --epochs 30` | 19,890.0s(약 5.52시간) | 30epoch, batch=32, lr 오류로 결과 폐기 |
| ResNet 학습(lr=0.0001, 채택) | `train_resnet.py --epochs 30` | 20,774.1s(약 5.77시간) | 30epoch, batch=32, best epoch=25/30 |
| 특징 추출(FV-3 Algorithm1 + FV-4) | `extract_features.py` | 14.3초 | train/val/test 303명 전체, Variant3+ResNet 순전파 + Algorithm1(원소별 선택) 포함 |
| CCA 계산(n_components=211) | `run_cca.py` | 9.1초 | train 212개로 fit, val/test는 transform만 |
| GB 학습·평가(Eq6/Eq7 둘 다) | `train_gb.py` | 8.9초 | GradientBoostingClassifier 2회(concat/합산) 학습+평가 |

**관찰**: 신경망(ResNet) 학습은 5시간 이상 걸리는데, CCA/GB로 넘어간 이후 파이프라인
(특징 추출→CCA→GB)은 **전부 합쳐도 32초** — 신경망 학습이 전체 소요시간의 압도적
대부분을 차지함. WOA(200 iteration × 100 population × 20 독립실행 = 최대 40만 회
피트니스 평가, 매번 KNN 학습·평가 포함)는 이보다 오래 걸릴 것으로 예상되며, 실제
착수 시 소요시간을 이 표에 추가할 것.

콘솔 로그 원본(재현성 확인용): `extract_features.py`/`run_cca.py`/`train_gb.py` 실행
결과는 매번 코드 형태로 재현 가능(신경망과 달리 랜덤 시드 고정 시 완전 결정적) -
스크립트 자체가 `C:\Users\user\AppData\Local\Temp\claude\...\scratchpad\`에 보관돼
있으며, 필요시 프로젝트 폴더로 정식 이관 가능.

## 7. Table 6/7 고해상도 확인 완료

Table 6(Model-3/4 성능)과 Table 7(WOA 등 최적화 알고리즘 하이퍼파라미터)을 400dpi로
재렌더링해 정확히 확인함(예전 pdftotext 추출은 숫자가 뒤섞여 있었음).

### Table 6 — Model-3(WOA 전)/Model-4(WOA 후) 성능 (우리의 최종 목표치)

| | 5-fold | 10-fold | 15-fold |
|---|---|---|---|
| Model-3(CCA 융합만) Acc/Recall/Precision/F1 | 91.6/88.6/93.5/90.5 | 92.4/88.2/92.8/91.8 | 94.8/90.4/95.6/93.8 |
| Model-4(WOA 적용) Acc/Recall/Precision/F1 | 95.9/90.7/98.6/92.9 | 96.2/89.3/98.8/93.4 | 97.2/90.6/98.5/93.1 |

### Table 7 — WOA 하이퍼파라미터(확정)

| 파라미터 | 값 |
|---|---|
| Constant(b, 나선상수) | 1 |
| Threshold(thres) | 0.5 |
| Population Size | 100 |
| Number of iterations | 200 |
| Lower/upper bound | 0, 1 |
| 독립 실행 횟수 | 20 |

(PSO/GSA/GA/ACO 등 비교 대상 알고리즘 파라미터도 같은 표에 있으나 우리 프로젝트
범위 밖이라 생략 - 필요시 `table6_7-13.png` 참조)

## 8. WOA 특징선택 구현 및 결과

### 8-1. 알고리즘 요약

논문 Eq(8)~(17) 그대로 구현(`woa_feature_selection.py`). 해(고래)는 CCA 융합 결과(Eq7
합산, 211차원)에 대응하는 길이 211의 이진 벡터(1=특징 선택). 매 iteration마다 고래
100마리가 "포위(encircling)/탐색(search)/나선공격(bubble-net)" 셋 중 하나로 위치를
갱신하고, 연속값을 sigmoid+threshold(0.5)로 이진화. 적합도(Eq.17)는
`alpha*KNN오류율 + beta*(선택특징수/전체특징수)`로, 선택된 특징만으로 KNN(k=5)을
train(212)에 학습시켜 val(45)로 오류율을 잼(test는 오염 방지를 위해 최종 평가에만 사용).

**자체결정 값**(논문 미기재): alpha=0.99, KNN k=5(2026-07-03 세션 `DEVIATIONS.md` 기록과
동일), 이진화 방식(sigmoid+threshold), R(D) 평가에 test 대신 val 사용.

### 8-2. 결과 — 20회 독립실행

| 지표 | 평균 | 최고 | 최저 | 표준편차 |
|---|---|---|---|---|
| Test Accuracy | 40.76% | 58.70%(run5, 특징 36개) | 28.26% | 7.86%p |
| Test F1(macro) | 38.33% | 52.69% | 28.39% | - |

총 소요시간: 2744.0초(약 45.7분), 실행당 평균 137.2초.

**중요한 발견 — 논문과 반대 경향**: 20회 평균(40.76%)이 WOA 적용 전(Eq7 합산 전체
211개 특징, 45.65%)보다 **오히려 낮습니다.** 최고 실행(58.70%)만 보면 개선된 것
같지만, 이전에 "여러 시드 중 최선만 골라 보고하면 안 된다"고 정한 원칙(같은 이유로
평균을 봐야 함 - `용어설명_시드와_클래스붕괴.md` 참조)을 그대로 적용하면, **평균
기준으로는 WOA가 오히려 성능을 낮췄습니다.** 논문(Model3→Model4, 91.6%→95.9% 등
항상 개선)과 정반대 경향. 원인 후보:
- 적합도 평가에 쓰는 val셋이 45개뿐이라 노이즈가 커서, WOA가 "진짜 좋은" 특징이
  아니라 val셋에 우연히 맞는 특징을 고르는 방향으로 수렴했을 가능성(이 프로젝트에서
  반복적으로 나타난 "작은 데이터셋 → 지표 노이즈" 패턴의 또 다른 사례)
  - alpha=0.99, KNN k=5 등 자체결정값이 논문의 실제 설정과 달라서 최적화 방향
  자체가 어긋났을 가능성
- CCA 융합 특징 자체가 이미 논문 대비 품질이 낮은 상태(45.65%, 논문은 91%+)라 그 위에
  추가 최적화를 해도 밑바탕이 부실해 개선 여지가 적었을 가능성

### 8-3. 원인 실증 분석 (2026-07-22 추가)

`results/woa_20runs_result.json`을 직접 분석해 위 원인 후보 중 어느 것이 실제로
맞는지 확인함.

**(1) 수렴 자체는 정상**: 20회 실행 모두 초기 fitness(0.67, 무작위 초기화 수준)에서
최종 0.11~0.31까지 확실히 감소 — WOA 최적화 루프 자체는 논문 수식대로 잘 작동함.

**(2) 결정적 문제 — fitness와 test accuracy가 사실상 무관함**: 20개 실행의
fitness값과 test accuracy 사이 상관계수 = **+0.243**(양의 상관). fitness는 낮을수록
좋게 설계했으므로(오류율↓+선택특징수↓), 정상이라면 fitness가 낮을수록 test
accuracy가 높아야(음의 상관) 하는데 오히려 미약한 양의 상관 — **"val셋에서 좋다고
판단한 특징"과 "실제 test에서 잘 일반화되는 특징"이 서로 안 맞는다는 뜻.** 원인
후보 1번(45개짜리 작은 val셋의 노이즈)이 실증적으로 뒷받침됨.

**(3) 특징 선택 자체가 불안정**: 211개 특징 중 20회에 걸쳐 최소 한 번이라도 선택된
것이 174개(82%)나 됨 - 즉 "이게 확실히 중요하다"고 반복적으로 뽑히는 소수 특징
그룹이 없고, 거의 전체 특징 공간을 이리저리 헤매는 모습. 평균 선택빈도는 20회 중
5.2회(26%)뿐. **안정적으로 수렴하는 진짜 신호가 없다는 뜻** — 이 역시 데이터
부족(212개)으로 CCA 융합 특징 자체의 신호 대 잡음비가 낮은 것과 일관됨.

**결론**: WOA 구현 자체는 버그가 아니라 논문 수식대로 정확히 작동하고 있음. 문제는
**적합도 평가에 쓰는 val셋(45개)이 너무 작아 노이즈가 심하다는, 이 프로젝트 전체를
관통하는 근본적 한계**(작은 데이터셋 → 지표 양자화·불안정, `용어설명_시드와_
클래스붕괴.md` 참조)가 WOA 단계에서도 그대로 재현된 것.

### 8-4. 더 큰 질문 — 논문은 같은 규모(303명) 데이터로도 왜 잘 나오는가

WOA 단계까지 오면서 격차가 계속 확인되니, "데이터 규모는 같은데 왜 논문만 잘 되는가"를
정리해봄. 확신도(이번 세션에서 직접 확인된 것 vs 추정)로 나눔.

**직접 확인됨(methodology 차이) — 가장 유력**:
1. **k-fold 교차검증 vs 단일 holdout**: 논문 Table5가 CNN/ResNet 둘 다 "k-fold=5"를
   명시(이번 세션에서 재확인). 저희는 처음부터 끝까지 **단일 70/15/15 holdout**만
   써왔음. k-fold는 (a)폴드마다 학습 데이터가 더 많고(242~273개 vs 저희 212개
   고정), (b)여러 분할의 평균이라 "운 나쁜 분할 하나"에 덜 휘둘림 — 이 프로젝트
   내내 반복 관찰된 "시드/분할 바뀌면 결과가 크게 흔들림" 문제 자체를 k-fold
   평균이 완화시켜 줌.
2. **그리드서치 결과=여러 조합 중 최댓값**: Table3 Study2는 pooling×activation×
   batch×flatten×optimizer×lr 조합을 다 탐색한 뒤 **가장 잘 나온 조합 하나만
   보고**. 오늘 WOA 20회 실행에서 평균(40.76%)과 최고(58.70%) 차이가 18%p나 났던
   것처럼, "여러 시도 중 최선만 보고"하면 그 자체로 낙관 편향(selection bias)이
   생김 — 논문 숫자가 이미 이런 종류의 편향을 안고 있을 가능성.
3. **데이터 증강 미구현**: 논문 Methods에 "data augmentation" 언급이 있음(구체
   기법 미기재). 저희는 구현 안 함 - 212개 학습 샘플에 증강 없이는 모델이 보는
   실질적 다양성이 논문보다 훨씬 적음.

**의심되지만 확정 못 함**:
4. **전처리 파이프라인 재구성 오차**: 논문 자체가 전처리 서술을 일관되지 않게
   해서(Methods/Results 불일치) 저희 재구성이 실제 파이프라인과 다를 가능성 -
   PD 진단의 핵심 신호(nigrosome-1 sign)가 수 mm급으로 미세해서 전처리 정밀도
   차이에 민감할 수 있음(육안검증 결과 참조:
   `세션_기록_Variant1_교정_및_조기종료_버그수정.md` 5절).
5. **논문 서술만으로는 못 찾은 추가 세부사항**: 이번 세션에서만도 배치사이즈
   오독, Variant1 구조 오판, FC-3/FC-4 차원 3번 번복 등 숱한 모호함을 발견함 -
   지금 시점에도 확인 못 한 세부사항이 더 있을 가능성이 높음(논문 원저자의 실제
   코드에만 있는 디테일).

**추측(검증 안 됨, 가능성만 언급)**:
6. 원저자 파이프라인 자체에 미세한 데이터 누수(leakage)가 있었을 가능성 - 예를 들어
   정규화 통계를 분할 전 전체 데이터로 계산했거나, subject 단위가 아닌 image 단위로
   k-fold를 나눴다면(동일 피험자의 다른 스캔이 train/test에 동시에 들어감) 실제보다
   낙관적인 숫자가 나올 수 있음. 다만 이건 논문에서 확인할 방법이 없는 순수 추측임.

**정리**: 가장 설명력이 큰 건 1번(k-fold vs holdout)과 2번(그리드서치 최댓값 보고) -
둘 다 "같은 모델이라도 평가 방법론 자체가 다르면 숫자가 크게 벌어질 수 있다"는
점을 보여주는 사례이고, 이번 세션에서 직접 관찰한 사실(Table5 k-fold 명시, WOA
평균-최고 격차 18%p)로 뒷받침됨.

## 9. 결과 로그 구조 (`results/woa_20runs_result.json`)

사용자 요청으로 로그 기록 방식 자체를 문서화. JSON 최상위 구조:

```
{
  "hyperparams": {population, iterations, n_independent_runs, b_const, threshold,
                   alpha, beta, knn_k},          # Table7 확정값 + 자체결정값 전부 기록
  "total_elapsed_sec": 2744.0,                    # 20회 전체 소요시간
  "summary": {test_accuracy_mean/max/min/std,     # 20회 분포 요약(평균이 대표값)
              test_f1_mean/max/min, best_run_index},
  "runs": [                                        # 20개 원소, 실행 1개당 1개
    {
      "run": 0, "seed": 1000,                      # 실행 번호/시드(재현용)
      "n_selected": 47,                            # 선택된 특징 수
      "selected_indices": [3, 7, 12, ...],          # 선택된 CCA 성분의 인덱스(재현·분석용)
      "fitness": 0.1782,                            # 이 실행의 최종 최적 fitness(낮을수록 좋음)
      "convergence": [0.35, 0.31, ..., 0.1782],     # iteration별 best fitness 추이(201개,
                                                     # 초기값+200iteration - 수렴 곡선 그리기용)
      "test_accuracy": 0.3478, "test_precision_macro": ..., "test_recall_macro": ...,
      "test_f1_macro": 0.3162,                      # 이 실행이 찾은 특징으로 최종 GB 학습 후 test 성능
      "elapsed_sec": 148.1                          # 이 실행 1회 소요시간
    },
    ... (총 20개)
  ]
}
```

이 구조로 (1)수렴이 실제로 됐는지(`convergence` 배열이 우하향하는지), (2)실행마다
결과가 얼마나 다른지(`runs`의 `test_accuracy` 분산), (3)어떤 특징이 반복적으로
선택되는지(`selected_indices`를 20개 실행에 걸쳐 집계) 등을 나중에 재분석할 수 있음.

관련 파일: `woa_feature_selection.py`(스크립트), `results/woa_20runs_result.json`(결과).

## 10. Model-3 k-fold(5/10/15) 평가 — 방식 A(특징 추출기 고정)

### 10-1. 방식

CNN(Variant3)·ResNet은 **재학습 없이** 기존 고정 체크포인트(70/15/15 holdout으로
1회 학습됨)를 특징 추출기로만 재사용. `dataset.py`의 `get_kfold_splits(k)`(subject
단위 stratified k-fold, 2026-07-03 세션에 이미 구현돼 있던 것 발견해 재사용)로
303명을 k개 fold로 나누고, **fold마다 CCA(Eq7 합산)+GB만 새로 학습·평가**.
`kfold_model3.py`로 구현.

**구현 중 발견한 문제**: n_components를 fold 크기 최댓값(최대 282)까지 쓰면
sklearn CCA의 SVD가 간헐적으로 수렴 실패 — 211부터 시작해 실패하면 20씩 줄여
재시도하도록 방어 코드 추가(fold 15개 중 실제로 1개 fold에서 발동, 성공까지
재시도됨).

### 10-2. 결과 — 논문과 반대 경향

| k | 우리 Accuracy | 논문 Accuracy | 단일holdout(45.65%) 대비 |
|---|---|---|---|
| 5 | 39.2%(±6.7%p) | 91.6% | **-6.4%p** |
| 10 | 36.5%(±?) | 92.4% | **-9.1%p** |
| 15 | 38.5%(±7.1%p) | 94.8% | **-7.1%p** |

**k-fold가 단일 holdout보다 오히려 나쁩니다.** 처음 세운 가설("k-fold를 쓰면 논문처럼
더 안정적이고 좋은 값이 나올 것")과 정반대 결과.

클래스별 F1(fold 전체 평균) — PD가 fold 수가 늘수록 더 나빠짐:

| k | Control | Prodromal | PD |
|---|---|---|---|
| 5 | 49.2% | 26.3% | 33.1% |
| 10 | 49.4% | 26.3% | 22.2% |
| 15 | 52.6% | 34.9% | **16.3%** |

합산 confusion matrix(k=10 기준)에서 PD 94개 중 74개가 Control로 오분류 — 특정
클래스 쏠림이 뚜렷함.

### 10-3. 원인 분석 — "특징 추출기 고정 + 분류기만 k-fold"의 구조적 한계

CNN/ResNet이 원래 학습에 썼던 212명(원래 train)이 이번 k-fold의 **test fold에도
섞여 들어갈 수 있음** — 그 사람들 특징은 모델이 이미 "본" 데이터라 부당하게
좋게 나오고, 원래 val/test(91명)였던 사람이 test fold에 들어가면 "정직하게"
평가됨. 즉 **fold마다 특징의 신뢰도 자체가 뒤섞여 있어 오히려 노이즈가 커짐**.

논문 Table5가 "CNN/ResNet 자체도 k=5"라고 명시한 건, 아마 **fold마다 CNN/ResNet부터
다시 학습**시켰다는 뜻으로 추정됨(그래야 이런 오염이 없음) — 이번 방식(방식 A)은
그 조건을 만족하지 못하는 단순화된 버전이었음.

### 10-4. 방식 B(fold마다 재학습) — 계획만 수립, 아직 미실행

"제대로" 하려면 fold마다 CNN(Variant3)/ResNet을 처음부터 재학습해야 함. 예상 비용
(1회 학습 시간: Variant3=20.6분, ResNet=5.77시간 기준):

| | k=5만 | k=5+10+15(총 30fold) |
|---|---|---|
| Variant3 | 103분(1.7시간) | 618분(**10.3시간**) |
| ResNet | **28.9시간(1.2일)** | **173시간(약 7.2일)** |

ResNet만으로도 k=5 하나에 하루 이상, Table6과 똑같이 5+10+15를 다 맞추려면
일주일 이상 걸림 — 이번 세션에서는 시간상 미실행. 실행 여부는 이 비용을 보고
별도 판단 필요.

## 11. 남은 작업

- **방식 B(fold마다 CNN/ResNet 재학습) 실행 여부 결정** — 위 10-4 비용표 참조,
  사용자 판단 필요
- WOA가 평균적으로 성능을 낮춘 원인 조사(8-3에서 fitness-accuracy 무상관까지는
  확인, 근본 해결은 아직) - 미해결
- Model-4(WOA+GB) k-fold 평가 — 방식 A로도 아직 미실행(Model-3만 함)
- Figure 4(WOA 흐름도), Figure 5/6(최적화 알고리즘 비교, 수렴곡선) 아직 시각적으로 미확인
- `train.py`(구버전, `D:\Brain_Tensor` 경로 참조하는 낡은 오케스트레이션 스크립트)와
  `full_run_result.json`(2026-07-16자, 이번 세션 이전 결과) 정리 필요 - 삭제/보관
  여부 사용자 확인 필요
