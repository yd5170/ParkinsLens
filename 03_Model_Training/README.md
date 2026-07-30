# 03_Model_Training

**Model-1(3D-CNN)과 Model-2(3D-ResNet)의 신경망 가중치를 실제로 학습시키는 단계만**
다룹니다. CCA 융합(Model-3)·WOA 특징선택(Model-4)·GB 분류기 평가는 여기 없습니다
(각각 `04_Feature_Engineering`, `05_Model_Evaluation` 참고) - 2026-07-24에 폴더를
"학습 vs 특징엔지니어링 vs 평가"로 명확히 분리함.

## 먼저 볼 것

[`00_문서_읽는_순서.md`](00_문서_읽는_순서.md)에 전체 문서/코드 읽는 순서가 정리돼
있습니다. 새 하이퍼파라미터/구조를 정할 때는 [`DEVIATIONS.md`](DEVIATIONS.md)를
먼저 검색하세요(논문 미기재 항목을 왜 이렇게 정했는지 근거가 다 있음).

## 구성

| 파일 | 내용 |
|---|---|
| `train_ablation.py` | Base/Variant1/Variant2/Variant3 학습. `--kfold 5\|10\|15`로 Table5의 k-fold CV 재현 가능(`run_kfold()`). 학습 루프(`train_loop()`)는 epoch별 loss/accuracy/precision/recall/f1(macro+weighted)까지 기록하고, `train_cnn3d.py`/`train_resnet.py`와 공유됨 |
| `train_cnn3d.py` | 최종모델 `CNN3D`(Figure2 직접 재현) 학습 |
| `train_resnet.py` | `ResNet3D` 학습(`--kfold` 지원) |
| `dataset.py` | `get_holdout_split()`(70/15/15 hold-out), `get_kfold_splits()`(Model-3/4 평가용, train/test만), `get_kfold_splits_with_val()`(CNN/ResNet 학습 자체의 k-fold용, train/val/test 3분할) |
| `results/` | 학습 결과 JSON(모델별 하이퍼파라미터·epoch history·test 지표) |
| `checkpoints/` | 학습된 가중치(.pt) - `04_Feature_Engineering`의 특징 추출이 여기서 읽어감 |
| `reports/` | `04_Feature_Engineering`(구 버전)이 만든 Excel 요약 보고서 사본 |
| `archive/` | 로그 파일, 스테일 결과, 프로파일링 스크립트, 폐기된 구버전 CCA/WOA 스크립트 등 - 재사용 안 하고 이력 보관용으로만 둠 |

## 실행 방법

```bash
# 단일 holdout(기본)
python train_ablation.py --variant base       # or variant1/variant2/variant3
python train_cnn3d.py
python train_resnet.py

# 논문 Table5의 k-fold CV(5/10/15) 재현
python train_ablation.py --variant base --kfold 5
python train_resnet.py --kfold 5
```

GPU(RTX 4060/4070급) 환경 전제. `KMP_DUPLICATE_LIB_OK=TRUE` 환경변수가 필요할 수
있습니다(Anaconda MKL/OMP 중복 충돌 회피).

## 다음 단계

학습이 끝나면 체크포인트를 `04_Feature_Engineering/run_real_pipeline.py`가 읽어서
CCA 융합→WOA 특징선택까지 이어집니다.
