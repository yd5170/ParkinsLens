# 세션 기록 — 클래스 붕괴 대응 (Class Weight / WeightedRandomSampler)

## 배경

Variant3(24-layer, `ablation_models.py`의 `CNN3D_Variant3`)를 포함해 이 프로젝트에서 학습한
여러 모델이 클래스 붕괴(Class Collapse) 상태를 반복해서 보임 — 특히 Prodromal(가장 적은
클래스, train 212개 중 41개, 19%)을 아예 예측하지 않는 패턴.

데이터 분포: Control=110, Prodromal=58, PD=135 (전체) / Train 212(Control 77, Prodromal 41,
PD 94), Val 45, Test 46.

## 시도 1 — Class-weighted Loss (2026-07-25)

- **위치**: `train_ablation.py`의 `compute_class_weights()`(신규 함수), `train_loop()`의
  `criterion` 생성부, `main()`의 `--class_weighted` 플래그.
- **방식**: train_loader의 실제 라벨 분포로 역빈도 가중치 계산
  (`weight_c = total / (n_classes * count_c)`) → `nn.CrossEntropyLoss(weight=...)`에 적용.
  **학습(train) 시에만 적용**하고 val/test 평가 loss는 비가중 그대로 유지(평가 지표의
  일관성 유지 목적).
- **적용 클래스 가중치**(train 분포 [77, 41, 94] 기준): Control=0.9177, Prodromal=1.7236,
  PD=0.7518.
- **기본값 off** — `--class_weighted`를 명시적으로 줘야 적용됨. 기존 실행(Base/V1/V2 등)의
  재현성을 깨지 않기 위함.

### 결과 (Variant3, 30epoch, patience=999)

| | class weight 없음 | class weight 적용 후 |
|---|---|---|
| Test Accuracy | 45.65% | 39.13% |
| Control recall | 0% (0/17) | 12% (2/17) |
| Prodromal recall | 0% (0/8) | **0% (0/8) — 그대로** |
| PD recall | 100% (21/21) | 76% (16/21) |

결과 JSON: `results/ablation_variant3_20260725_193450_acc39.1.json`
체크포인트: `checkpoints/ablation_variant3_20260725_193450_acc39.1.pt`

**판단**: Control 쪽은 완전 붕괴에서 벗어났지만 Prodromal은 여전히 0% — class weight 단독으로는
불충분. 전체 정확도도 오히려 하락(45.65%→39.13%)했지만, 이는 "PD만 찍어서 맞은 정확도"가
줄어든 것이라 정확도 하락 자체가 나쁜 신호는 아님(클래스 다양성은 늘었으므로).

## 시도 2 — WeightedRandomSampler 병행 (2026-07-25, 진행 중)

- **위치**: `train_ablation.py`의 `build_loaders_from_samples()`(`use_weighted_sampler` 인자
  추가), `build_loaders()`, `main()`의 `--sampler_weighted` 플래그.
- **방식**: train_loader 구성 시 `shuffle=True` 대신 `WeightedRandomSampler`(역빈도
  가중치, 복원추출)를 사용해 배치에 소수 클래스가 더 자주 뽑히도록 함. **val/test는
  실제 분포 그대로 유지**(평가 공정성 위해 손대지 않음).
- **class_weighted와 병행 적용** — 손실 가중치(class_weighted)만으로는 Prodromal recall이
  0%에서 안 움직였기 때문에, 샘플링 빈도 자체를 조정하는 방식을 추가로 시도.
- **기본값 off**, `--sampler_weighted` 명시 지정 시만 적용.

### 실행 커맨드

```
python train_ablation.py --variant variant3 --epochs 30 --patience 999 --class_weighted --sampler_weighted
```

### 발견한 버그와 수정 (첫 실행 시도 중, epoch1 진입 전 중단)

`compute_class_weights()`가 원래 `train_loader`를 배치 단위로 순회하며 라벨을 셌는데,
`--sampler_weighted`(WeightedRandomSampler)를 같이 켜면 loader를 순회하는 순간 **이미
재조정된 분포**가 나옴 — 실측: 원본 `[77,41,94]`인데 순회 결과는 `[70,77,65]`(거의 균등).
그 결과 class weight가 `[1.01, 0.92, 1.09]`로 사실상 1.0에 가까워져서 두 방법을 병행하는
의미가 없어짐.

**수정**: `train_loader.dataset.samples`(PPMIT2Dataset이 들고 있는 원본 샘플 리스트, sampler
적용 여부와 무관)에서 직접 라벨을 세도록 변경. 수정 후 재확인한 결과 원본 분포
`[77,41,94]` → 가중치 `Control=0.9177 Prodromal=1.7236 PD=0.7518`로 정상 계산됨을
확인하고 재학습 시작.

### 결과 (2026-07-26 완료)

`--class_weighted --sampler_weighted --checkpoint_metric val_acc`(체크포인트 기준도 팀원 J와
동일하게 val_acc 최고 + val_loss tie-break로 변경) 조합으로 재학습.

| | Control(17) | Prodromal(8) | PD(21) |
|---|---|---|---|
| Recall | 65% (11/17) | **38% (3/8)** | 43% (9/21) |
| Precision | 46% | 100% | 47% |

Confusion Matrix(rows=true, cols=pred): Control→[11,0,6], Prodromal→[1,3,4], PD→[12,0,9]

- Test Accuracy **50.00%**, F1(macro) **51.07%** — 이 프로젝트의 모든 Variant3 시도 중 최고 기록.
- **처음으로 3개 클래스 전부 recall>0%** — Prodromal도 8개 중 3개를 맞힘. 이전까지는 class
  weight만 적용해도 Prodromal은 항상 0%였음.
- 세 가지 변경(class weight + sampler + checkpoint 기준)을 한 번에 적용해서 어느 것이
  결정적이었는지는 이 실험만으로는 구분 불가 — 필요하면 각 요인을 하나씩 끄고 재실행해
  기여도를 분리할 수 있음(추후 검토).

결과 JSON: `results/ablation_variant3_20260726_035445_acc50.0.json`
체크포인트: `checkpoints/ablation_variant3_20260726_035445_acc50.0.pt`

## 관련 파일

- `train_ablation.py` — 실제 구현 위치(코드 내 `[2026-07-25 추가]` 주석 참조)
- `results/ablation_variant3_*.json` — 실행별 결과(hyperparams에 `class_weighted`,
  `class_weights`, `sampler_weighted` 기록됨)
- `Variant3_상세구조_결과비교_팀원J.xlsx` — 팀원 J 구현체와의 구조/결과 비교
