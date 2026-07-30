# 05_Model_Evaluation

`04_Feature_Engineering`이 만든 CCA 융합(Model-3)/WOA 선택(Model-4) 특징을 최종
ML 분류기(Gradient Boosting)로 학습·평가하는 단계입니다. 2026-07-24에
`03_Model_Training`에서 이 폴더로 이동·정리했습니다.

## 구성

| 파일 | 내용 |
|---|---|
| `train_gb.py` | `04_Feature_Engineering/run_real_pipeline.py`가 저장한 `fused_{train,val,test}.npz`를 읽어 Eq6(concat) vs Eq7(합산) 각각 GB로 학습·평가 |
| `kfold_model3.py` | Model-3(CCA+GB, WOA 적용 전)을 k-fold(5/10/15)로 평가 - 논문 Table6 재현. CNN/ResNet은 고정 체크포인트를 특징 추출기로만 재사용(재학습 없음), CCA는 fold마다 새로 fit(데이터 누출 방지) |
| `compute_confusion.py` | `03_Model_Training`의 체크포인트(.pt)를 재추론해서 confusion matrix, macro/weighted precision·recall·f1을 계산. `analyze_split(ckpt_path, split="train"\|"val"\|"test")`로 세 분할 다 지원. ablation 5종 + resnet3d + cnn3d 전부 지원 |
| `ml_classifiers_kfold_eval.py` | 2026-07-03 세션 작성. SVM/KNN/GB/RF 4개 분류기(`get_classifiers()`)와 k-fold 평가 하니스(`kfold_evaluate()`) - **weighted 평균** 사용 |

## ⚠️ 미해결 - macro vs weighted 평균 통일 필요

`ml_classifiers_kfold_eval.py`는 weighted 평균을 쓰는데, `compute_confusion.py`를
비롯해 이 프로젝트 대부분의 리포트는 macro 평균을 씁니다(클래스 불균형 Control
110/Prodromal 58/PD 135 상황에서 두 값이 꽤 다를 수 있음). 어느 쪽을 공식 기준으로
삼을지 아직 결정 안 됨 - 사용자 확인 필요.

## 논문에 없어서 자체 결정한 값

- GB 하이퍼파라미터: 논문 전혀 미기재 → scikit-learn 기본값 사용
- k-fold 평가 시 분류기를 GB로 고정(논문 본문 서술상 GB가 대체로 최고 성능이라 판단)

## 실행 방법

```bash
# 04_Feature_Engineering/run_real_pipeline.py를 먼저 실행해서 fused_*.npz를 만든 뒤
python train_gb.py
python kfold_model3.py
python ml_classifiers_kfold_eval.py
```
