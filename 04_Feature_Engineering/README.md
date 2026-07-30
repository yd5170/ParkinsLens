# 04_Feature_Engineering

**Model-3(CCA 융합)과 Model-4(WOA 특징선택) 단계**입니다. 신경망을 학습시키는 게
아니라, `03_Model_Training`에서 이미 학습된 CNN(Variant3)/ResNet 체크포인트의
중간출력(특징)을 수학적으로 융합·선택하는 것 - "학습"이 아니라 "특징 엔지니어링"이라
`03_Model_Training`과 분리해뒀습니다.

## ⚠️ 2026-07-24: 주논문 원문을 다시 정확히 대조해서 이 폴더 코드를 정정했습니다

이 폴더는 2026-07-03 세션에 처음 작성됐는데, 이후 세션에서 03_Model_Training에
별도로 CCA/WOA를 다시 만들면서 두 갈래로 갈라졌습니다. 주논문
(`../07_Document/주논문_nature.pdf`)을 다시 정확히 읽어 어느 쪽이 맞는지 확인한
결과, **이 폴더 쪽에 있던 실수를 아래처럼 정정하고, 이제 이 폴더가 유일한
CCA/WOA 기준 구현입니다** (03_Model_Training에 있던 중복 스크립트는
`03_Model_Training/archive/`로 이동).

| 항목 | 정정 전 | 정정 후(현재) | 논문 근거 |
|---|---|---|---|
| FV-3(CNN 융합특징) 차원 | 2000(단순 concat, Algorithm1 미구현) | **1000**(Algorithm1 원소별 max 선택) | p.6 "FV1×1000, FV2×1000", p.7 "2000 features from **both**" |
| CCA n_components | 임의 상한 100 | **수학적 최댓값**(min(n_samples-1, D_a, D_b)) | 논문에 상한 근거 없음 - CCA 자체의 수학적 한계를 그대로 씀 |
| CCA 융합 방식 | concat(Eq6)만 | **Eq6(concat)+Eq7(합산) 둘 다 계산** | 논문 "concatenated **or summed up**"(p.7) |
| WOA 이진화 방식 | 위치값 직접 threshold 비교 | (재검토 중 sigmoid로 바꿨다가 결함 발견 후) **다시 직접 비교로 원복** | Table7 "Lower and upper bound = 0 and 1" |
| WOA fitness 평가 | (없었음, 새로 추가) | **3-fold 교차검증**(`cross_val_score`) | 논문 전반의 교차검증 관행과 일관 + 실측상 baseline 대비 WOA가 실제로 개선됨을 확인(03_Model_Training/results/04pipeline_real_data_20runs_result.json) |

## 구성

| 파일 | 내용 | 논문 근거 |
|---|---|---|
| `extract_features.py` | Variant3/ResNet 체크포인트에서 FC-1/FC-2/FC-4 raw 출력 추출 → Algorithm1(원소별 max 선택, `algorithm1_feature_concatenation()`)로 FV-3(1000차원) 생성 | Algorithm 1 의사코드, Eq.2 |
| `cca_feature_fusion.py` | `cca_fuse(fv_a, fv_b)`: CCA로 두 특징(FV-3, FC-4)을 융합, Eq6(concat)/Eq7(합산) 둘 다 반환 | Eq.3-7 |
| `woa_feature_selection.py` | `binary_woa_feature_selection()`: Table7 하이퍼파라미터 그대로(population=100, iterations=200, bounds=[0,1], 독립실행 20회) | Eq.8-17, Table 7 |
| `run_real_pipeline.py` | 위 세 개를 실제 체크포인트 데이터로 이어붙여 Model-3/Model-4 전체를 한 번에 실행 - `fused_{train,val,test}.npz`와 최종 결과 JSON 저장 | - |

## 논문에 없어서 자체 결정한 값

- WOA fitness 가중치 alpha=0.99(정확도에 절대적 가중치), fitness 계산용 KNN K=5
- WOA fitness 평가 프로토콜(3-fold CV) - 위 표 참조

자세한 근거는 각 파일 상단 docstring 참고.

## 실행 방법

```bash
# 더미 데이터로 각 모듈 단독 동작만 빠르게 확인
python cca_feature_fusion.py
python woa_feature_selection.py

# 실제 체크포인트로 Model-3/4 전체 파이프라인 실행 (WOA 20회 독립실행 포함,
# 3-fold CV 특성상 1회당 약 9~10분, 전체 약 3시간 소요)
python run_real_pipeline.py
```

## 다음 단계

`run_real_pipeline.py`가 저장한 `fused_*.npz`를 `05_Model_Evaluation/train_gb.py`가
읽어서 GB 분류기로 최종 평가합니다.
