# -*- coding: utf-8 -*-
"""
train_ablation.py - Ablation Study(Table 3, Study 1) 재현용 학습 스크립트

02_Model_Definition/ablation_models.py 의 CNN3D_Base(8-layer, 논문 보고 정확도 82.02%)를
실제 303명 PPMI T2 데이터(data_final_303.csv)로 학습하고, Train/Val/Test accuracy와
Precision/Recall/F1을 출력합니다.

[실행 환경]
- 이 스크립트는 로컬 GPU(RTX 4060 등) 환경에서 실행하는 것을 전제로 작성되었습니다.
  개발 샌드박스에는 PyTorch가 설치되어 있지 않아, 여기서는 shape 계산만 검증했고
  (ablation_models.py 참조) 실제 forward pass/학습 실행은 검증하지 못했습니다.

[2026-07 갱신: 논문 Table 3(Study 2, Base model 열)에서 Base 모델 하이퍼파라미터가
실제로 명시되어 있음을 확인함 - 이전 기록은 "미기재"로 잘못 판단했던 것]
[2026-07-21 재정정] 위 최초 확인 당시 배치사이즈를 32로 읽었으나, 표를 다시 고해상도로
렌더링해 재확인한 결과 실제 체크 표시는 **Base=64**였음(Variant3가 32, 서로 반대로 잘못
읽었던 것). 아래는 정정된 값:
- 논문 Table 3 Study 2 표에서 Base model 열의 체크 표시를 재확인한 결과:
  Pooling=Max, Activation=ReLU, **Batch size=64**, Flatten=Flatten(Global max 아님),
  Optimizer=Adam, **Learning rate=0.01**, Epoch=30 로 전부 명시되어 있음.
  이 오독으로 그동안 Base 실험이 전부 batch=32로 잘못 돌아갔음 - batch=64로 정정 후
  재실행 필요(VARIANT_DEFAULT_HP 참조).
- 정확도 비교 목표치도 2가지로 구분됨:
  · Study 1(구조 탐색, 하이퍼파라미터 튜닝 전) 결과: 82.02%
  · Study 2(위 하이퍼파라미터 튜닝 후) 결과: 84.96%, Precision 87.34%, Recall 81.54%, F1 85.32%
  이 스크립트는 Study 2의 튜닝된 하이퍼파라미터를 사용하므로, 비교 기준은 84.96%가 더 적절함
  (82.02%는 튜닝 전 구조만 비교한 수치라 하이퍼파라미터가 다름).

[논문에 여전히 기재되지 않아 프로젝트에서 자체 결정한 값]
- 데이터 분할은 dataset.py의 get_holdout_split()을 그대로 재사용합니다
  (70/15/15, subject 단위 stratified, seed=42 고정 - 프로젝트 전체 일관성 유지).
- Weight decay(L2): Table 3 Study 2에는 항목 자체가 없어 여전히 미기재. Table 5(Variant3
  학습 설정)의 값(0.0001)을 그대로 가져와 사용 - 근거는 약하지만 프로젝트 일관성 유지 목적.

[2026-07 추가: base variant 첫 실행에서 로짓/로스 폭주(epoch1 loss 수천대) +
극심한 과적합(train_acc 100%, test_acc 34.78% vs 논문 Study2 목표 84.96%) 확인 후 대응]
- Gradient clipping(max_norm=1.0): 논문에 전혀 언급 없음(있는지/없는지조차 미기재).
  CNN3D_Base는 flatten(1,404,928) -> Linear(3) 단일 거대 classifier가 파라미터의
  98.7%를 차지해, lr=0.01(논문 Study2 명시값) 하에서 Adam 업데이트만으로도 로짓이
  쉽게 폭주함(관측: epoch1 val_loss=8172). lr/batch/epoch 등 논문 명시 하이퍼파라미터는
  그대로 두고, 학습 안정화를 위한 구현 디테일만 추가 - weight_decay와 동일한 성격의
  "논문 미기재 자체 결정" 항목.
- Best-validation-checkpoint 선택: 논문 Table 5에 원문 그대로 "Selected the optimal
  hyperparameters based on best validation performance"라고 명시되어 있음에도, 기존
  구현은 마지막 epoch(30) 가중치를 그대로 test에 사용했음 - 이는 오히려 논문 절차와
  어긋남. Val accuracy가 가장 높았던 epoch의 가중치를 저장해두었다가 그 체크포인트로
  test를 평가하도록 수정 - "논문 재현"의 정확도를 높이는 방향이며 하이퍼파라미터
  변경이 아님.

[2026-07-19 추가: gradient clipping/체크포인트 선택 적용 후에도 시드만 바꾸면 Control/
Prodromal 등 서로 다른 클래스가 번갈아 recall 0으로 붕괴하는 불안정성이 계속 관측됨
(예: seed=42 Control 붕괴 acc 56.52%, seed=43 Prodromal 붕괴 acc 50.00%) - 근본 원인
추가 조치]
- Classifier 출력 스케일 보정(calibrate_classifier_scale): He(Kaiming) 초기화는 입력이
  서로 독립(i.i.d.)이라는 가정 하에 분산을 계산하는데, CNN3D_Base의 classifier는
  conv+pool을 거친 공간적으로 상관관계가 큰 1,404,928차원 벡터를 그대로 입력받아 이
  가정이 깨짐 - 그 결과 초기 로짓 분산이 이론값보다 훨씬 커서 epoch1 loss가 수천대로
  폭주하고(이전 항목에서 관측), 이게 학습 궤적을 시드에 매우 민감하게 만드는 원인 중
  하나로 추정됨. 학습 시작 직후 실제 train 배치 하나로 classifier 출력의 표준편차를
  측정해서 목표값(1.0)에 맞게 classifier.weight/bias를 사후 스케일링 - 모델 구조
  (Flatten->Linear->SoftMax, 논문 명시)나 lr/batch/epoch 등 하이퍼파라미터는 전혀
  바꾸지 않고, 이미 자체 결정 사항이었던 초기화 스킴(He 초기화)만 보정하는 구현
  디테일. --no_calibrate_init으로 끌 수 있음(기본 활성).
"""
import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# [2026-07-21 추가] stdout이 파일로 리다이렉트(백그라운드 실행 등)되면 기본적으로 완전
# 버퍼링되어, epoch 하나에 3~4분씩 걸리는 이 스크립트 특성상 진행상황이 실시간으로 전혀
# 안 보이던 문제 - 01_Preprocessing/preparing_ref21order_v1.py와 동일한 방식으로
# 라인 버퍼링으로 전환.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# 형제 폴더(02_Model_Definition)에서 ablation_models.py를 임포트하기 위한 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)  # D:\Brain_Tensor
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_models import CNN3D_Base, CNN3D_Variant1, CNN3D_Variant1_Untuned, CNN3D_Variant2, CNN3D_Variant3
from dataset import get_holdout_split, get_kfold_splits_with_val, PPMIT2Dataset, augment_volume_3d

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


VARIANT_MODELS = {
    "base": CNN3D_Base,           # 8-layer, 논문 보고 82.02%
    "variant1": CNN3D_Variant1,   # 9-layer(conv 3개), Study2(튜닝후) 논문 보고 88.25%
    # [2026-07-21 추가] Study1(튜닝 전) 재현용 - 사용자 결정: "튜닝 전/후"를 구분해서
    # 보자는 요청에 따라, 구조(conv 3개)는 Variant1과 같지만 하이퍼파라미터는 그리드서치
    # 결과(Average/Global max/64/0.001) 대신 전 variant 공통 기본값(Max/Flatten/64/0.01)
    # 사용. Study1 목표치는 85.75%.
    "variant1_untuned": CNN3D_Variant1_Untuned,
    "variant2": CNN3D_Variant2,   # 17-layer(conv 2+3개, pooling 2회), 논문 보고 88.76%
    "variant3": CNN3D_Variant3,   # 24-layer(conv 2+3+2개, pooling 3회), 논문 보고 89.43%
}

# [2026-07-21 정정] Table 3 Study 2 표를 고해상도로 렌더링해 다시 육안 확인한 결과,
# 이전에 "Base 열 확인 결과 배치=32"라고 기록해뒀던 게 잘못 읽은 것이었음이 드러남.
# 실제 체크 표시는 Base=64, Variant1=64, Variant2=64, Variant3=32(그동안 알던 것과
# Base/Variant3가 서로 반대). Pooling/Flatten/LR은 기존 기록이 맞았음:
# Base(Max/64/Flatten/0.01), Variant1(Average/64/Global max/0.001),
# Variant2(Max/64/Flatten/0.01), Variant3(Average/32/Global max/0.001, 2026-07-21 구현 완료).
# 이 오류로 이번 세션(및 이전 세션) 내내 Base 실험이 batch=32로 잘못 돌아갔음 -
# batch=64로 정정 후 재실행 필요. Variant2는 "Base와 동일값"으로 잠정 가정했던 게
# 결과적으로 Pooling/Flatten/LR은 맞았지만 batch만 32로 잘못 가정돼 있었음.
VARIANT_DEFAULT_HP = {
    "base": {"batch_size": 64, "lr": 0.01},
    "variant1": {"batch_size": 64, "lr": 0.001},
    "variant1_untuned": {"batch_size": 64, "lr": 0.01},
    "variant2": {"batch_size": 64, "lr": 0.01},
    "variant3": {"batch_size": 32, "lr": 0.001},
}


def build_loaders_from_samples(train_samples, val_samples, test_samples, image_dir, batch_size,
                                use_weighted_sampler=False, use_augmentation=False):
    """샘플 리스트(dict)를 직접 받아 DataLoader 3개를 만듦. build_loaders()의
    csv holdout 분할 부분과 k-fold 분할(run_kfold) 양쪽에서 공통으로 재사용.

    [2026-07-25 추가] use_weighted_sampler=True면 train_loader만 WeightedRandomSampler로
    바꿔 소수 클래스(Prodromal)가 더 자주 뽑히도록 함 - class_weighted(손실 가중)와 병행
    적용 목적(둘 다 역빈도 기반이지만 하나는 loss 가중, 하나는 샘플링 빈도 자체를 조정).
    val/test는 실제 분포 그대로 평가해야 하므로 대상에서 제외.

    [2026-07-26 추가] use_augmentation=True면 train_ds에 dataset.augment_volume_3d()를
    on-the-fly transform으로 적용(회전/뒤집기/스케일/노이즈, dataset.py 참조) - 이미
    구현돼 있었지만 어떤 학습 스크립트도 실제로 안 쓰고 있었음. ResNet3D 과적합
    대응(Dropout과 병행) 목적으로 처음 실사용. val/test는 항상 원본 그대로 평가."""
    exists = lambda s: os.path.exists(os.path.join(image_dir, f"{s['sample_id']}.nii.gz"))
    train_samples = [s for s in train_samples if exists(s)]
    val_samples = [s for s in val_samples if exists(s)]
    test_samples = [s for s in test_samples if exists(s)]

    train_transform = (lambda x: augment_volume_3d(x, np.random.default_rng())) if use_augmentation else None
    if use_augmentation:
        print("[Augmentation] train에만 augment_volume_3d 적용(회전/뒤집기/스케일/노이즈)")
    train_ds = PPMIT2Dataset(train_samples, image_dir, transform=train_transform)
    val_ds = PPMIT2Dataset(val_samples, image_dir, transform=None)
    test_ds = PPMIT2Dataset(test_samples, image_dir, transform=None)

    if use_weighted_sampler:
        train_labels = [int(s["label"]) for s in train_samples]
        n_classes = max(train_labels) + 1
        class_counts = [train_labels.count(c) for c in range(n_classes)]
        sample_weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[WeightedRandomSampler] train 분포={class_counts} -> 샘플링 확률을 역빈도로 재조정(복원추출)")
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[데이터] Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    return train_loader, val_loader, test_loader


def build_loaders(csv_path, image_dir, batch_size, seed, use_weighted_sampler=False, use_augmentation=False):
    train_samples, val_samples, test_samples = get_holdout_split(csv_path, seed=seed)
    return build_loaders_from_samples(train_samples, val_samples, test_samples, image_dir, batch_size,
                                       use_weighted_sampler=use_weighted_sampler,
                                       use_augmentation=use_augmentation)


@torch.no_grad()
def calibrate_classifier_scale(model, sample_x, device, target_std=1.0):
    """분류기(classifier) 레이어 출력 스케일 보정 - 스크립트 상단 docstring(2026-07-19
    항목) 참조. He 초기화는 입력이 서로 독립이라고 가정하지만 classifier 입력(flatten된
    conv 특징)은 공간적으로 상관되어 있어 실제 로짓 분산이 이론값보다 커짐. 실제 train
    배치 하나로 출력 표준편차를 재서 목표값에 맞게 classifier.weight/bias를 스케일링."""
    if not hasattr(model, "classifier"):
        return None, None
    model.eval()
    logits = model(sample_x.to(device))
    current_std = logits.std().item()
    if current_std > 1e-8:
        scale = target_std / current_std
        model.classifier.weight.mul_(scale)
        if model.classifier.bias is not None:
            model.classifier.bias.mul_(scale)
    model.train()
    return current_std, target_std


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip_norm=None, progress_every=50):
    """optimizer가 주어지면 학습 모드, 없으면 평가 모드로 동작.

    [2026-07-21 추가] epoch 하나가 batch=64 기준 3~4분씩 걸려 그 사이 콘솔이 오래 조용한
    문제 - train 단계에서 progress_every(기본 50)명 처리할 때마다 중간 진행상황을 찍음
    (val 단계는 샘플 수가 적고 금방 끝나 생략). 콘솔 출력일 뿐이며 training_log.txt/csv나
    결과 JSON에는 저장되지 않음(그쪽은 epoch 단위 요약만 기록 - history는 JSON에 남지만
    이 중간 진행상황 라인 자체는 어디에도 파일로 안 남음).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    next_progress_mark = progress_every

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

            if is_train and total >= next_progress_mark:
                running_acc = correct / total
                print(f"    [진행] {total}명 처리 (batch_loss={loss.item():.4f}, "
                      f"누적train_acc={running_acc:.4f})")
                next_progress_mark += progress_every

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, all_preds, all_labels


def evaluate_final(model, loader, device, criterion, split_name):
    loss, acc, preds, labels = run_epoch(model, loader, device, criterion, optimizer=None)
    print(f"\n=== {split_name} 최종 평가 ===")
    print(f"  Loss={loss:.4f}  Accuracy={acc:.4f} ({acc*100:.2f}%)")

    result = {"loss": loss, "accuracy": acc}
    if _HAS_SKLEARN and len(set(labels)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        print(f"  Precision(macro)={precision:.4f}  Recall(macro)={recall:.4f}  F1(macro)={f1:.4f}")
        print(classification_report(labels, preds, target_names=["Control", "Prodromal", "PD"], zero_division=0))
        result.update({"precision_macro": precision, "recall_macro": recall, "f1_macro": f1})
    return result


def _macro_prf(labels, preds):
    """[2026-07-23 추가] epoch마다 precision/recall/f1(macro)도 기록해달라는 요청 -
    지금까지는 val_acc/val_loss만 history에 남겨서, 친구 쪽 training_history_ResNet.csv
    (train_f1_macro/val_precision_macro/val_recall_macro/val_f1_macro 컬럼 있음)와
    epoch 단위로 비교할 수가 없었음. run_epoch()이 이미 preds/labels를 반환하므로
    추가 forward pass 없이 그대로 계산 가능."""
    if not _HAS_SKLEARN or len(set(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return p, r, f1


def _weighted_prf(labels, preds):
    """[2026-07-24 추가] "없음"으로 비어있던 Epoch Metrics/Best Validation의 Weighted
    지표 칸을 채우기 위해 macro와 동일한 방식으로 weighted(support 비례 가중) 버전도
    계산. 친구 쪽 리포트(cnn_training_report (1).xlsx)에 이미 있는 컬럼과 형식을 맞춤."""
    if not _HAS_SKLEARN or len(set(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return p, r, f1


def _pred_counts(preds, n_classes=3):
    """[2026-07-24 추가] 친구 리포트의 "Validation Prediction Counts"(클래스별 예측
    개수)에 대응 - 클래스 붕괴 여부를 한눈에 보여줌(예: 전부 한 클래스로만 쏠렸는지).
    class_names(["Control","Prodromal","PD"]) 순서에 맞춰 인덱스 리스트로 반환."""
    counts = [0] * n_classes
    for p in preds:
        counts[p] += 1
    return counts


def compute_class_weights(train_loader, n_classes, device):
    """[2026-07-25 추가] 클래스 불균형(Control=110/Prodromal=58/PD=135) 완화용 역빈도
    가중치(inverse frequency, weight_c = total / (n_classes * count_c)).

    [2026-07-25 수정] 처음엔 train_loader를 배치 단위로 순회하며 라벨을 셌는데,
    --sampler_weighted(WeightedRandomSampler)와 함께 쓰면 loader를 순회하는 순간 이미
    "재조정된" 분포가 나와서(실측: 원본 [77,41,94] -> 순회 결과 [70,77,65]) 가중치가
    거의 1.0으로 무의미해지는 버그를 실행 중 발견함(첫 재학습 시도, epoch1 진입 전
    중단하고 수정). loader.dataset.samples(=PPMIT2Dataset이 들고 있는 원본 샘플
    리스트)에서 직접 세도록 변경 - sampler 적용 여부와 무관하게 항상 실제 원본 분포를
    반영함."""
    counts = [0] * n_classes
    for s in train_loader.dataset.samples:
        counts[int(s["label"])] += 1
    total = sum(counts)
    weights = [total / (n_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def train_loop(model, train_loader, val_loader, device, lr, weight_decay, epochs,
               grad_clip_norm, patience, min_epochs, calibrate_init, log_prefix="", class_weights=None,
               checkpoint_metric="val_loss"):
    """[2026-07-23 추가] main()의 학습 루프(epoch 순회 + best-validation-loss 체크포인트
    선택 + 조기종료)를 함수로 뽑아냄 - 기존 단일 holdout 실행과 신규 run_kfold() 양쪽에서
    동일 로직을 재사용하기 위함(k-fold라고 학습 방식 자체가 달라지면 안 되므로 완전히
    같은 코드 경로를 사용). 반환값은 main()이 그대로 써온 변수들과 1:1 대응.

    [2026-07-25 추가] checkpoint_metric="val_loss"(기본, 기존 동작 유지) 또는 "val_acc".
    논문 Table 5는 "best validation performance" 기준이라고만 되어 있고 accuracy/loss
    어느 쪽인지 명시가 없음(원문 직접 확인 완료, 표에 Validation Accuracy 열 자체가 없음) -
    이 프로젝트는 val_loss(연속값이라 45개뿐인 val set에서도 민감하게 반응, 2026-07-20
    변경 사유 참조)를, 팀원 J는 val_accuracy(동점시 val_loss로 tie-break)를 씀 - 둘 다
    이 모호한 문구에 대한 서로 다른 해석. "val_acc"를 주면 팀원과 동일한 기준(최고
    val_acc, 동점시 최저 val_loss)으로 비교 실험 가능."""
    if calibrate_init:
        sample_x, _, _ = next(iter(train_loader))
        before_std, target_std = calibrate_classifier_scale(model, sample_x, device)
        if before_std is not None:
            print(f"{log_prefix}[초기화 보정] classifier 출력 표준편차: {before_std:.2f} -> {target_std:.2f}로 스케일 조정")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    no_improve_count = 0
    t0 = time.time()
    for epoch in range(epochs):
        epoch_t0 = time.time()
        train_loss, train_acc, train_preds, train_labels = run_epoch(
            model, train_loader, device, criterion, optimizer, grad_clip_norm=grad_clip_norm
        )
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, device, criterion, optimizer=None)
        train_p, train_r, train_f1 = _macro_prf(train_labels, train_preds)
        val_p, val_r, val_f1 = _macro_prf(val_labels, val_preds)
        train_pw, train_rw, train_f1w = _weighted_prf(train_labels, train_preds)
        val_pw, val_rw, val_f1w = _weighted_prf(val_labels, val_preds)
        val_counts = _pred_counts(val_preds)
        epoch_elapsed = time.time() - epoch_t0
        total_elapsed_so_far = time.time() - t0
        now_str = time.strftime("%H:%M:%S")
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "train_precision_macro": train_p, "train_recall_macro": train_r, "train_f1_macro": train_f1,
            "train_precision_weighted": train_pw, "train_recall_weighted": train_rw, "train_f1_weighted": train_f1w,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision_macro": val_p, "val_recall_macro": val_r, "val_f1_macro": val_f1,
            "val_precision_weighted": val_pw, "val_recall_weighted": val_rw, "val_f1_weighted": val_f1w,
            "val_pred_counts": val_counts,
            "epoch_seconds": round(epoch_elapsed, 2), "timestamp": now_str,
        })
        print(f"{log_prefix}[{now_str}] [Epoch {epoch+1}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({train_acc*100:.2f}%) train_f1={train_f1:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({val_acc*100:.2f}%) val_f1={val_f1:.4f}  "
              f"소요시간={epoch_elapsed:.1f}s (누적 {total_elapsed_so_far:.1f}s)")

        if checkpoint_metric == "val_acc":
            # 팀원 J와 동일 기준: val_acc가 더 높으면 갱신, 같으면 val_loss가 더 낮을 때만 갱신(tie-break)
            is_better = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        else:
            is_better = val_loss < best_val_loss

        if is_better:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if epoch + 1 >= min_epochs and no_improve_count >= patience:
                metric_desc = f"val_acc={best_val_acc:.4f}" if checkpoint_metric == "val_acc" else f"val_loss={best_val_loss:.4f}"
                print(f"{log_prefix}[조기 종료] {checkpoint_metric}가 {patience}epoch 연속 개선되지 않아 "
                      f"{epoch+1}epoch에서 학습 중단 - best epoch {best_epoch} 가중치 사용 (best {metric_desc})")
                break

    elapsed = time.time() - t0
    actual_epochs = len(history)
    model.load_state_dict(best_state)
    return {
        "history": history, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss, "best_state": best_state,
        "elapsed": elapsed, "actual_epochs": actual_epochs,
    }


def run_kfold(args):
    """[2026-07-23 추가] 논문 Table5("k-fold CV=5")를 CNN/ResNet 학습 자체에 반영.
    fold마다 train_loop()로 독립 학습(같은 하이퍼파라미터, 같은 seed 기반 재현성) 후,
    fold별 결과 + 평균/표준편차를 하나의 집계 JSON으로 저장함. 기존 단일 holdout 결과
    JSON과 형식을 최대한 맞춰(summary/history 구조 유지) 리포트 빌더가 그대로 읽을 수
    있게 함. get_kfold_splits_with_val()의 val 분리 방식(nested holdout)은 dataset.py
    docstring 참조 - 논문 미기재, 자체 결정."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_cls = VARIANT_MODELS[args.variant]
    fold_summaries = []
    fold_records = []
    kfold_t0 = time.time()

    for fold_idx, (train_samples, val_samples, test_samples) in enumerate(
        get_kfold_splits_with_val(args.csv_path, k=args.kfold, seed=args.seed)
    ):
        fold_no = fold_idx + 1
        print(f"\n{'='*20} [Fold {fold_no}/{args.kfold}] variant={args.variant} {'='*20}")
        torch.manual_seed(args.seed + fold_idx)  # fold마다 다른 초기화(재현성은 seed+fold로 유지)

        train_loader, val_loader, test_loader = build_loaders_from_samples(
            train_samples, val_samples, test_samples, args.image_dir, args.batch_size,
            use_weighted_sampler=getattr(args, "sampler_weighted", False),
        )

        model = model_cls(num_classes=3).to(device)
        n_params = sum(p_.numel() for p_ in model.parameters())

        fold_class_weights = None
        if getattr(args, "class_weighted", False):
            fold_class_weights, fold_counts = compute_class_weights(train_loader, n_classes=3, device=device)
            print(f"  [Fold {fold_no}] 클래스 가중치(역빈도, 분포={fold_counts}): "
                  f"Control={fold_class_weights[0]:.4f} Prodromal={fold_class_weights[1]:.4f} PD={fold_class_weights[2]:.4f}")

        loop_result = train_loop(
            model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, grad_clip_norm=args.grad_clip_norm, patience=args.patience,
            min_epochs=args.min_epochs, calibrate_init=args.calibrate_init,
            log_prefix=f"  [Fold {fold_no}] ", class_weights=fold_class_weights,
        )

        criterion = nn.CrossEntropyLoss()
        test_result = evaluate_final(model, test_loader, device, criterion, f"Fold {fold_no} Test")
        test_result["best_epoch"] = loop_result["best_epoch"]
        test_result["best_val_acc"] = loop_result["best_val_acc"]

        fold_summary = {
            "fold": fold_no,
            "n_params": n_params,
            "actual_epochs": loop_result["actual_epochs"],
            "best_epoch": loop_result["best_epoch"],
            "best_val_acc": round(loop_result["best_val_acc"], 4),
            "test_accuracy": round(test_result["accuracy"], 4),
            "test_precision_macro": round(test_result.get("precision_macro", float("nan")), 4),
            "test_recall_macro": round(test_result.get("recall_macro", float("nan")), 4),
            "test_f1_macro": round(test_result.get("f1_macro", float("nan")), 4),
            "elapsed_sec": round(loop_result["elapsed"], 1),
        }
        fold_summaries.append(fold_summary)
        fold_records.append({
            "fold": fold_no,
            "summary": fold_summary,
            "history": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                for row in loop_result["history"]
            ],
        })
        print(f"  [Fold {fold_no} 결과] Acc={fold_summary['test_accuracy']*100:.2f}% "
              f"Precision={fold_summary['test_precision_macro']*100:.2f}% "
              f"Recall={fold_summary['test_recall_macro']*100:.2f}% "
              f"F1={fold_summary['test_f1_macro']*100:.2f}%  best_epoch={fold_summary['best_epoch']}")

        # fold별 체크포인트도 저장 (특징추출 등에 필요할 수 있어 단일 holdout과 동일하게 유지)
        checkpoints_dir = os.path.join(_ROOT, "03_Model_Training", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        ts_fs = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(
            checkpoints_dir,
            f"ablation_{args.variant}_kfold{args.kfold}_fold{fold_no}_{ts_fs}_acc{fold_summary['test_accuracy']*100:.1f}.pt",
        )
        torch.save({
            "variant": args.variant, "model_state_dict": loop_result["best_state"],
            "num_classes": 3, "input_size": 56, "class_names": ["Control", "Prodromal", "PD"],
            "kfold": args.kfold, "fold": fold_no,
            "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                             "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
                             "calibrate_init": args.calibrate_init, "seed": args.seed},
            "test_accuracy": test_result["accuracy"],
            "test_precision_macro": test_result.get("precision_macro"),
            "test_recall_macro": test_result.get("recall_macro"),
            "test_f1_macro": test_result.get("f1_macro"),
            "best_epoch": loop_result["best_epoch"],
        }, ckpt_path)

    kfold_elapsed = time.time() - kfold_t0

    def _mean_std(key):
        vals = [fs[key] for fs in fold_summaries if fs[key] == fs[key]]  # NaN 제외(자기 자신과 다름 판정)
        if not vals:
            return float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, var ** 0.5

    agg = {}
    for key in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]:
        mean, std = _mean_std(key)
        agg[key + "_mean"] = round(mean, 4)
        agg[key + "_std"] = round(std, 4)

    print(f"\n{'='*20} [K-Fold(k={args.kfold}) 집계 결과 - {args.variant}] {'='*20}")
    print(f"Accuracy : {agg['test_accuracy_mean']*100:.2f}% ± {agg['test_accuracy_std']*100:.2f}%p")
    print(f"Precision: {agg['test_precision_macro_mean']*100:.2f}% ± {agg['test_precision_macro_std']*100:.2f}%p")
    print(f"Recall   : {agg['test_recall_macro_mean']*100:.2f}% ± {agg['test_recall_macro_std']*100:.2f}%p")
    print(f"F1       : {agg['test_f1_macro_mean']*100:.2f}% ± {agg['test_f1_macro_std']*100:.2f}%p")
    print(f"총 소요시간: {kfold_elapsed:.1f}s ({kfold_elapsed/3600:.2f}시간)")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fs = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{agg['test_accuracy_mean']*100:.1f}"
    out_path = os.path.join(results_dir, f"kfold_{args.variant}_k{args.kfold}_{timestamp_fs}_acc{acc_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "timestamp": timestamp, "variant": args.variant, "kfold": args.kfold,
                "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                                 "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
                                 "calibrate_init": args.calibrate_init, "seed": args.seed,
                                 "patience": args.patience, "min_epochs": args.min_epochs},
                "n_folds": args.kfold, "elapsed_sec": round(kfold_elapsed, 1),
                **agg,
            },
            "fold_summaries": fold_summaries,
            "fold_records": fold_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"K-Fold 집계 결과 JSON 저장: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Ablation Study 모델 학습 + 정확도 출력")
    p.add_argument("--variant", choices=list(VARIANT_MODELS.keys()), default="base")
    # [2026-07-20 갱신] 기본값이 존재하지 않는 경로(data_final_303.csv, 전처리06_리사이즈_최종)를
    # 가리키고 있던 걸 발견 - 이번 세션 내내 --csv_path/--image_dir를 매번 직접 지정해서
    # 드러나지 않았음. 여러 전처리 변형(min-max/N4없음, N4있음, z-score)을 비교한 결과
    # min-max+N4없음(전처리_ref21order_v1)이 가장 우수·안정적이어서 이걸 표준으로 채택
    # (상세: 01_Preprocessing/전처리_ref21order_v1_상세기록.md, 전처리_변형_종합비교.md).
    p.add_argument("--csv_path", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv"))
    p.add_argument("--image_dir", type=str, default=os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1"))
    p.add_argument("--epochs", type=int, default=30)
    # [2026-07-20 변경] variant마다 논문 그리드서치 최적값이 다르므로(VARIANT_DEFAULT_HP
    # 참조) 여기서는 고정 기본값을 주지 않고 None으로 둔 뒤, args.variant가 정해진
    # 다음에 VARIANT_DEFAULT_HP에서 채워 넣음. --batch_size/--lr을 직접 지정하면 그 값이
    # 우선 적용됨(그리드서치 재현이 아닌 별도 실험을 하고 싶을 때 사용).
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    # [2026-07 추가] 논문 미기재, 학습 안정화를 위한 자체 결정값 - 스크립트 상단 docstring 참조
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    # [2026-07-20 추가] 사용자 요청: val_acc가 N epoch 연속 개선되지 않으면(과적합 신호)
    # 남은 epoch을 다 채우지 않고 조기 종료. 어차피 test 평가는 항상 best-validation
    # 체크포인트를 쓰므로(위 Table 5 인용부 참조) 결과 값 자체는 바뀌지 않고, 더 이상
    # 개선 안 되는 뒤쪽 epoch을 도는 시간만 절약됨.
    p.add_argument("--patience", type=int, default=5,
                   help="validation loss가 이 횟수만큼 연속으로 개선 안 되면 조기 종료")
    # [2026-07-20 추가] Variant1(batch=64, train 212개 -> epoch당 4배치)처럼 epoch당
    # 업데이트 횟수가 적은 조합에서 patience만으로는 너무 이르게(실제 사례: best epoch
    # 1~2, 6~7epoch만에 종료) 멈추는 문제가 있어 최소 학습 epoch 수를 강제함.
    p.add_argument("--min_epochs", type=int, default=15,
                   help="이 epoch 수를 채우기 전에는 조기 종료를 허용하지 않음")
    p.add_argument(
        "--no_calibrate_init", action="store_false", dest="calibrate_init", default=True,
        help="classifier 초기화 출력 스케일 보정 비활성화(기본은 활성) - 스크립트 상단 docstring 참조",
    )
    # [2026-07-23 추가] 논문 Table5("k-fold CV=5")를 실제로 반영해달라는 요청 - 지금까지는
    # 이 항목을 "그리드서치 내부 검증용"으로 해석해 단일 70/15/15 holdout만 실행해왔음
    # (DEVIATIONS.md 6번 섹션 참조). --kfold 5(또는 10/15)를 주면 get_kfold_splits_with_val()
    # 로 fold마다 별도 학습을 돌리고, 결과를 하나의 집계 JSON(fold별 값 + 평균/표준편차)으로
    # 저장함. 지정하지 않으면(기본 None) 기존 단일 holdout 방식 그대로 동작.
    p.add_argument("--kfold", type=int, default=None, choices=[5, 10, 15],
                   help="지정 시 Table5의 k-fold CV를 재현(fold마다 학습, 결과 집계). 미지정시 기존 단일 holdout")
    # [2026-07-25 추가] 클래스 붕괴(소수 클래스 recall 0%) 대응 - 논문에 없는 자체 결정
    # 항목이므로 기본값 off로 두고, 명시적으로 지정했을 때만 적용(기존 결과 재현성 유지).
    p.add_argument("--class_weighted", action="store_true", default=False,
                   help="train_loader의 실제 클래스 분포로 역빈도 가중치를 계산해 CrossEntropyLoss에 적용(학습 시에만, test/val 평가 loss는 비가중)")
    # [2026-07-25 추가] class_weighted(손실 가중)만으로는 Prodromal recall이 0%에서 안 움직여
    # 병행 시도 - WeightedRandomSampler로 train 샘플링 자체를 역빈도로 재조정(val/test는
    # 실제 분포 그대로 유지, 평가 공정성 위해 손대지 않음). 기본 off, 명시적 지정 시만 적용.
    p.add_argument("--sampler_weighted", action="store_true", default=False,
                   help="train_loader를 WeightedRandomSampler(역빈도, 복원추출)로 구성 - class_weighted와 병행 적용 가능")
    # [2026-07-25 추가] 체크포인트 선택 기준 비교 실험용 - 논문 Table5("best validation
    # performance")가 accuracy/loss 어느 쪽인지 명시하지 않아 이 프로젝트는 val_loss를
    # 써왔는데, 팀원 J는 val_acc(동점시 val_loss)를 씀 - 같은 기준으로 바꿔서 비교해보기 위함.
    p.add_argument("--checkpoint_metric", type=str, default="val_loss", choices=["val_loss", "val_acc"],
                   help="best-checkpoint 선택 기준. val_loss(기본, 기존 동작) 또는 val_acc(팀원 J와 동일, 동점시 val_loss로 tie-break)")
    args = p.parse_args()

    default_hp = VARIANT_DEFAULT_HP[args.variant]
    if args.batch_size is None:
        args.batch_size = default_hp["batch_size"]
    if args.lr is None:
        args.lr = default_hp["lr"]

    if args.kfold:
        run_kfold(args)
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = build_loaders(
        args.csv_path, args.image_dir, args.batch_size, args.seed,
        use_weighted_sampler=args.sampler_weighted,
    )

    model_cls = VARIANT_MODELS[args.variant]
    model = model_cls(num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[모델] {args.variant} ({model_cls.__name__}), 파라미터 수={n_params:,}")

    class_weights = None
    class_weight_values = None
    if args.class_weighted:
        class_weights, class_counts = compute_class_weights(train_loader, n_classes=3, device=device)
        class_weight_values = [round(w, 4) for w in class_weights.tolist()]
        print(f"[클래스 가중치] train 분포(Control/Prodromal/PD)={class_counts} -> "
              f"가중치 Control={class_weight_values[0]:.4f} Prodromal={class_weight_values[1]:.4f} PD={class_weight_values[2]:.4f}")

    # [2026-07-23 리팩터] classifier 초기화 스케일 보정은 train_loop() 내부에서 수행됨
    # (calibrate_init=args.calibrate_init로 전달) - 여기서 별도로 호출하면 두 번 적용되는
    # 버그가 생기므로 제거.
    # 아래 학습 루프(epoch 순회 + best-validation-loss 체크포인트
    # 선택 + 조기종료 + precision/recall/f1 기록)는 train_loop()로 추출됨 - run_kfold()도
    # 동일 함수를 호출하므로 단일 holdout/k-fold 두 경로가 완전히 같은 학습 로직을 씀.
    # [2026-07-20 변경] best/조기종료 판단 기준을 val_acc -> val_loss로 변경. val 샘플이
    # 45개뿐이라 val_acc는 1/45(~2.2%p) 단위로만 움직이는 양자화된 지표라 여러 epoch
    # 연속으로 완전히 같은 값이 나오기 쉬움(실제로 seed42/43 모두 5epoch 내내 고정값 관찰).
    # val_loss는 연속값이라 같은 상황에서도 학습 진행 여부를 훨씬 민감하게 반영함.
    loop_result = train_loop(
        model, train_loader, val_loader, device, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, grad_clip_norm=args.grad_clip_norm, patience=args.patience,
        min_epochs=args.min_epochs, calibrate_init=args.calibrate_init, class_weights=class_weights,
        checkpoint_metric=args.checkpoint_metric,
    )
    history = loop_result["history"]
    best_epoch = loop_result["best_epoch"]
    best_val_acc = loop_result["best_val_acc"]
    best_val_loss = loop_result["best_val_loss"]
    best_state = loop_result["best_state"]
    elapsed = loop_result["elapsed"]
    actual_epochs = loop_result["actual_epochs"]
    print(f"\n총 학습 시간: {elapsed:.1f}s")

    # 논문 Table 5("Selected the optimal hyperparameters based on best validation
    # performance")를 따라, 마지막 epoch가 아닌 best-validation epoch의 가중치로 test 평가
    print(f"\n[체크포인트 선택] Best validation epoch = {best_epoch}/{actual_epochs} "
          f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, {best_val_acc*100:.2f}%) - 이 시점 가중치로 test 평가")

    criterion = nn.CrossEntropyLoss()
    test_result = evaluate_final(model, test_loader, device, criterion, "Test")
    test_result["best_epoch"] = best_epoch
    test_result["best_val_acc"] = best_val_acc

    # ============================================================
    # 논문 Table 3 정보 (Study 1 구조 정보 + Study 2 하이퍼파라미터 + 4개 평가지표)
    # 논문 값과 우리 모델 값을 한 표에서 나란히 비교할 수 있도록 구성
    # ============================================================
    # [2026-07-20 추가] Variant1(9-layer, conv 3개) 등록. Study1 구조/정확도(85.75%)는
    # 논문 원문 직접 인용으로 확정("three 3D-conv layers... 9-layer architecture...
    # achieving a model accuracy of 85.75%", ablation_models.py CNN3D_Variant1 참조).
    # Study2 accuracy/precision/recall/f1(88.25/89.93/84.28/87.43) 및 나머지 그리드서치
    # 항목(Pooling/Flatten/Batch/LR)은 Table3 Study2 표에서 Variant1 열을 육안으로 직접
    # 확인 완료(아래 "variant1" 딕셔너리 주석 참조) - 더 이상 잠정 가정 아님.
    # [2026-07-20 추가] Variant2(17-layer, conv 2+3개, pooling 2회) 등록. Study1
    # 구조/정확도(88.76%)는 논문 원문 직접 인용으로 확정("two 3D-conv layers...
    # three 3D-conv layers... 17-layer architecture... achieving an accuracy of
    # 88.76%", ablation_models.py CNN3D_Variant2 참조 - conv/pooling 개수 표기는
    # 논문 Table3 Study1 원문의 "2+3"/"1+1" 표기 그대로 사용). Study2
    # accuracy/precision/recall/f1(90.28/91.83/87.72/90.81)은 모델_아키텍처_분석.md
    # Table3 인용부 확인.
    # [2026-07-21 정정] Table3 Study2 표를 고해상도로 렌더링해 Base/Variant1/Variant2/
    # Variant3 4개 열 전부 재확인. Base의 batch_size를 32로 잘못 읽고 있었던 게 드러남
    # (실제는 64, Variant3가 32 - 서로 반대). Variant2의 batch_size도 "Base와 동일"
    # 가정으로 32를 썼었는데 이 역시 64로 정정(Pooling=Max/Flatten=Flatten/LR=0.01은
    # 기존 가정이 맞았음 - 육안 확인으로 최종 확정).
    paper_architecture = {
        "base": {"n_conv_layers": "2", "n_pooling_layers": "1", "test_acc": 0.8202, "finding": "Lowest accuracy"},
        "variant1": {"n_conv_layers": "3", "n_pooling_layers": "1", "test_acc": 0.8575, "finding": "Intermediate"},
        # variant1_untuned은 conv/pooling 개수(구조)가 variant1과 동일 - Study1 목표치도 동일하게 사용
        "variant1_untuned": {"n_conv_layers": "3", "n_pooling_layers": "1", "test_acc": 0.8575, "finding": "Intermediate"},
        "variant2": {"n_conv_layers": "2+3", "n_pooling_layers": "1+1", "test_acc": 0.8876, "finding": "Intermediate"},
        # [2026-07-21 추가] Variant3(24-layer, conv 2+3+2개, pooling 3회). 사용자가
        # 제공한 "Proposed 3D CNN" 다이어그램으로 구조 확정(ablation_models.py
        # CNN3D_Variant3 참조) - conv 블록 2+3+2, Study1 목표 89.43%("Highest").
        "variant3": {"n_conv_layers": "2+3+2", "n_pooling_layers": "1+1+1", "test_acc": 0.8943, "finding": "Highest"},
    }
    paper_study2 = {
        "base": {
            "pooling": "Max", "activation": "ReLU", "batch_size": 64, "flatten": "Flatten",
            "optimizer": "Adam", "lr": 0.01, "epochs": 30,
            "accuracy": 0.8496, "precision": 0.8734, "recall": 0.8154, "f1": 0.8532,
        },
        # [2026-07-20 정정] Table3 Study2 Variant1 열을 육안으로 직접 확인 - 이전엔
        # Base와 동일값으로 잠정 가정했으나 전부 다른 값이었음이 확인됨:
        # Pooling=Average(Base=Max), Flatten=Global max(Base=Flatten, 구조 자체가
        # 다름 - ablation_models.py CNN3D_Variant1 참조), Batch=64(Base도 64로 동일 -
        # 2026-07-21 재확인), LR=0.001(Base=0.01). Activation=ReLU/Optimizer=Adam/
        # Epoch=30은 Base와 동일.
        "variant1": {
            "pooling": "Average", "activation": "ReLU", "batch_size": 64, "flatten": "Global max",
            "optimizer": "Adam", "lr": 0.001, "epochs": 30,
            "accuracy": 0.8825, "precision": 0.8993, "recall": 0.8428, "f1": 0.8743,
        },
        # [2026-07-21 육안 확인 완료] Pooling=Max/Flatten=Flatten/LR=0.01은 기존
        # 잠정 가정이 맞았고, batch_size만 64로 정정(32였던 건 오독).
        "variant2": {
            "pooling": "Max", "activation": "ReLU", "batch_size": 64, "flatten": "Flatten",
            "optimizer": "Adam", "lr": 0.01, "epochs": 30,
            "accuracy": 0.9028, "precision": 0.9183, "recall": 0.8772, "f1": 0.9081,
        },
        # [2026-07-21 추가] "튜닝 전"(Study1) 재현용 - 사용자 결정에 따라 전 variant
        # 공통 기본 하이퍼파라미터(Max/Flatten/64/0.01) 사용. 목표 정확도는 Study1의
        # 85.75%(논문이 Study1에서는 precision/recall/f1을 안 보고해 None으로 둠 -
        # 비교표에서 "N/A(Study1 미보고)"로 표시됨).
        "variant1_untuned": {
            "pooling": "Max", "activation": "ReLU", "batch_size": 64, "flatten": "Flatten",
            "optimizer": "Adam", "lr": 0.01, "epochs": 30,
            "accuracy": 0.8575, "precision": None, "recall": None, "f1": None,
        },
        # [2026-07-21 추가] Table3 Study2 Variant3 열 육안 확인 완료. Base/V1/V2와
        # 달리 batch_size=32(나머지 셋은 64) - 표에서 유일하게 다른 값.
        "variant3": {
            "pooling": "Average", "activation": "ReLU", "batch_size": 32, "flatten": "Global max",
            "optimizer": "Adam", "lr": 0.001, "epochs": 30,
            "accuracy": 0.9341, "precision": 0.9576, "recall": 0.8898, "f1": 0.9168,
        },
    }
    # 비교표 "실험값" 칸에 쓸, 실제 우리 모델의 구조 설명(변형마다 다름 - 이전엔
    # Base 문구가 하드코딩돼 있어 Variant1/2에도 잘못 표시되던 문제 수정)
    variant_arch_desc = {
        "base": {"conv": "2 (conv1, conv2)", "pool": "1 (MaxPool3d)",
                 "pool_type": "Max (MaxPool3d)", "flatten_type": "Flatten"},
        "variant1": {"conv": "3 (conv1~conv3)", "pool": "1 (AvgPool3d)",
                     "pool_type": "Average (AvgPool3d)", "flatten_type": "Global max (AdaptiveMaxPool3d)"},
        "variant1_untuned": {"conv": "3 (conv1~conv3)", "pool": "1 (MaxPool3d)",
                              "pool_type": "Max (MaxPool3d)", "flatten_type": "Flatten"},
        "variant2": {"conv": "2+3 (블록1 2개 + 블록2 3개)", "pool": "1+1 (블록마다 1개, MaxPool3d)",
                     "pool_type": "Max (MaxPool3d)", "flatten_type": "Flatten"},
        "variant3": {"conv": "2+3+2 (블록1 2개 + 블록2 3개 + 블록3 2개)",
                     "pool": "1+1+1 (블록마다 1개, AvgPool3d)",
                     "pool_type": "Average (AvgPool3d)",
                     "flatten_type": "Global max (AdaptiveMaxPool3d) + FC-1/FC-2"},
    }
    arch_desc = variant_arch_desc.get(args.variant, {"conv": "?", "pool": "?"})
    paper_acc_study1 = {k: v["test_acc"] for k, v in paper_architecture.items()}
    paper_acc_study2 = {k: v["accuracy"] for k, v in paper_study2.items()}
    paper_hyperparams = {k: {"epochs": v["epochs"], "batch_size": v["batch_size"], "lr": v["lr"]}
                          for k, v in paper_study2.items()}

    comparison_rows = []  # (항목, 논문 값, 우리 모델 값, 비고)

    if args.variant in paper_architecture:
        arch = paper_architecture[args.variant]
        s2 = paper_study2[args.variant]

        actual_precision = test_result.get("precision_macro", float("nan"))
        actual_recall = test_result.get("recall_macro", float("nan"))
        actual_f1 = test_result.get("f1_macro", float("nan"))

        # -- 구조(Study 1) --
        comparison_rows.append(("Conv layer 개수", str(arch["n_conv_layers"]), arch_desc["conv"], "일치"))
        comparison_rows.append(("Pooling layer 개수", str(arch["n_pooling_layers"]), arch_desc["pool"], "일치"))
        comparison_rows.append(("구조 단계 Test Accuracy (Study1, 튜닝 전)",
                                 f"{arch['test_acc']*100:.2f}%", "-", f"Finding: \"{arch['finding']}\""))

        # -- 하이퍼파라미터(Study 2) --
        comparison_rows.append(("Pooling 종류", s2["pooling"], arch_desc["pool_type"],
                                 "일치" if s2["pooling"] in arch_desc["pool_type"] else "다름"))
        comparison_rows.append(("Activation", s2["activation"], "ReLU",
                                 "일치" if s2["activation"] == "ReLU" else "다름"))
        comparison_rows.append(("Flatten 방식", s2["flatten"], arch_desc["flatten_type"],
                                 "일치" if s2["flatten"] in arch_desc["flatten_type"] else "다름"))
        comparison_rows.append(("Optimizer", s2["optimizer"], "Adam",
                                 "일치" if s2["optimizer"] == "Adam" else "다름"))
        comparison_rows.append(("Batch size", str(s2["batch_size"]), str(args.batch_size),
                                 "일치" if s2["batch_size"] == args.batch_size else "다름 - 확인 필요"))
        comparison_rows.append(("Learning rate", str(s2["lr"]), str(args.lr),
                                 "일치" if s2["lr"] == args.lr else "다름 - 확인 필요"))
        comparison_rows.append(("Epoch", str(s2["epochs"]), str(actual_epochs),
                                 "일치" if s2["epochs"] == actual_epochs else
                                 (f"조기 종료(요청 {args.epochs}epoch 중 {actual_epochs}epoch)" if actual_epochs < args.epochs
                                  else "다름 - 확인 필요")))

        # -- 평가지표 4종(Study 2 기준, 논문과 동일 하이퍼파라미터) --
        # [2026-07 변경] 차이(%p)는 표에 안 적고 실험값/논문값을 나란히 칸으로만 구분
        # [2026-07-21 추가] variant1_untuned(Study1 재현)처럼 논문이 precision/recall/f1을
        # 안 보고한 경우를 위한 N/A 처리
        fmt_pct = lambda v: f"{v*100:.2f}%" if v is not None else "N/A(Study1 미보고)"
        comparison_rows.append(("Accuracy", fmt_pct(s2["accuracy"]), f"{test_result['accuracy']*100:.2f}%", ""))
        comparison_rows.append(("Precision", fmt_pct(s2["precision"]), f"{actual_precision*100:.2f}%", ""))
        comparison_rows.append(("Recall", fmt_pct(s2["recall"]), f"{actual_recall*100:.2f}%", ""))
        comparison_rows.append(("F1-score", fmt_pct(s2["f1"]), f"{actual_f1*100:.2f}%", ""))

        # -- Finding 비교 (논문의 상대순위 vs 우리 모델의 목표 대비 판정) --
        acc_diff_pct = (test_result["accuracy"] - s2["accuracy"]) * 100
        if acc_diff_pct >= -3:
            our_finding = f"논문 목표치({s2['accuracy']*100:.2f}%)와 근접 - 정상 재현으로 판단"
        elif acc_diff_pct >= -10:
            our_finding = "논문 대비 다소 낮음 - 재현 오차 범위 내일 수 있으나 확인 권장"
        else:
            our_finding = "논문 대비 상당히 낮음 - 전처리/학습 파이프라인 점검 필요"
        comparison_rows.append(("Finding (논문 상대순위)", f"\"{arch['finding']}\" (Base/V1/V2/V3 4개 비교 기준)",
                                 "-", "우리는 Base만 실행 - 상대순위는 V1~V3까지 만들어야 확정 가능"))
        comparison_rows.append(("Finding (우리 모델, 자동판정)", "-", our_finding,
                                 f"기준: Study2 목표({s2['accuracy']*100:.2f}%) 대비 차이 {acc_diff_pct:+.2f}%p"))

        # -- 소요 시간(논문에 없음, 참고용) --
        comparison_rows.append(("총 소요 시간", "-", f"{elapsed:.1f}초 (약 {elapsed/60:.1f}분)", "논문 미기재"))
        comparison_rows.append(("Epoch당 평균 시간", "-", f"{elapsed/max(actual_epochs,1):.1f}초", "논문 미기재"))

        diff = test_result["accuracy"] - s2["accuracy"]

    # ---- 콘솔에 표 형태로 출력 ----
    # [2026-07 변경] 실험값 칸을 논문값보다 앞에 두고(요청: "실험값 논문값 이렇게"), 비고 칸은
    # 일치 여부/Finding처럼 값 자체가 아닌 주석에만 사용(수치 비교 행은 위에서 비고를 비움)
    print("\n========== [논문(Table 3) vs 우리 모델 비교표] ==========")
    if comparison_rows:
        col1_w = max(len(r[0]) for r in comparison_rows) + 2
        col2_w = max(len("실험값"), *(len(r[2]) for r in comparison_rows)) + 2
        col3_w = max(len("논문값"), *(len(r[1]) for r in comparison_rows)) + 2
        header = f"{'항목':<{col1_w}}{'실험값':<{col2_w}}{'논문값':<{col3_w}}비고"
        print(header)
        print("-" * len(header))
        for label, paper_v, model_v, note in comparison_rows:
            print(f"{label:<{col1_w}}{model_v:<{col2_w}}{paper_v:<{col3_w}}{note}")
    else:
        print("논문 보고값이 없는 모델입니다.")
    print("=" * 60)
    import csv

    # 로그 파일에 누적 기록 (TXT)
    log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_fs = time.strftime("%Y%m%d_%H%M%S")  # 파일명용(콜론 등 제외, JSON 파일명에 사용)
        # [2026-07-20 추가] epoch 수가 적은(<5) 실행은 정식 30-epoch 비교용이 아니라
        # 배선 확인용 스모크 테스트인 경우가 대부분 - 로그에서 바로 구분되도록 표시
        test_tag = " (스모크 테스트)" if args.epochs < 5 else ""
        # [2026-07-20 변경] 조기 종료된 경우 실제로 돈 epoch 수(actual_epochs)를 남기고,
        # 요청한 최대 epoch(args.epochs)와 다르면 "(조기 종료)"를 표시해 혼동 방지
        early_stop_tag = f" (조기 종료, 요청 {args.epochs})" if actual_epochs < args.epochs else ""
        f.write(f"\n[{timestamp}] Variant: {args.variant}{test_tag} | Epochs: {actual_epochs}{early_stop_tag} | Batch: {args.batch_size} | 총 소요시간: {elapsed:.1f}s\n")
        # [2026-07 변경] 실험값을 먼저 적고 논문값을 뒤에, 차이(%p)는 안 적음(요청 반영)
        if args.variant in paper_acc_study2:
            p_ep = paper_hyperparams[args.variant]["epochs"]
            p_bs = paper_hyperparams[args.variant]["batch_size"]
            diff = test_result["accuracy"] - paper_acc_study2[args.variant]
            f.write(f"  - [실험값] Epochs: {actual_epochs}{early_stop_tag}, Batch: {args.batch_size} -> Acc: {test_result['accuracy']*100:.2f}%\n")
            f.write(f"  - [논문값(Study2)] Epochs: {p_ep}, Batch: {p_bs}, LR: {paper_hyperparams[args.variant]['lr']} "
                    f"-> Acc: {paper_acc_study2[args.variant]*100:.2f}% (참고: Study1 튜닝전 82.02%)\n")
        else:
            f.write(f"  - [실험값] Acc: {test_result['accuracy']*100:.2f}%\n")
            diff = 0.0
        f.write("-" * 50 + "\n")
    print(f"로그 누적 기록 완료 (TXT): {log_path}")

    # 로그 파일에 누적 기록 (CSV)
    # [2026-07 변경] 실험값 컬럼을 논문값 컬럼들보다 앞에 두고, Diff 컬럼은 제거(요청 반영)
    csv_log_path = os.path.join(_ROOT, "03_Model_Training", "training_log.csv")
    csv_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["Timestamp", "Variant", "Actual Epochs", "Actual Batch Size", "Elapsed Time (s)",
                              "Actual Acc (%)", "Paper Acc Study2 (%)", "Paper Acc Study1 (%)"])

        paper_acc_s1_val = paper_acc_study1[args.variant] * 100 if args.variant in paper_acc_study1 else ""
        paper_acc_s2_val = paper_acc_study2[args.variant] * 100 if args.variant in paper_acc_study2 else ""
        actual_acc_val = test_result['accuracy'] * 100

        writer.writerow([
            timestamp,
            args.variant,
            actual_epochs,
            args.batch_size,
            round(elapsed, 1),
            round(actual_acc_val, 2),
            paper_acc_s2_val,
            paper_acc_s1_val,
        ])
    print(f"로그 누적 기록 완료 (CSV): {csv_log_path}")

    # [2026-07 추가] 기존에는 ablation_{variant}_result.json 고정 파일명이라 재실행 시
    # 이전 실행의 상세 결과(특히 best checkpoint 관련 history)가 덮어써져 사라졌음.
    # results/ 폴더에 실행 시각+정확도를 파일명에 포함시켜 실행마다 별도 파일로 누적 보존하고,
    # 파일을 열지 않아도 폴더 목록만으로 결과를 스캔할 수 있게 함.
    results_dir = os.path.join(_ROOT, "03_Model_Training", "results")
    os.makedirs(results_dir, exist_ok=True)
    acc_tag = f"{test_result['accuracy']*100:.1f}"
    out_path = os.path.join(results_dir, f"ablation_{args.variant}_{timestamp_fs}_acc{acc_tag}.json")

    # summary를 history보다 먼저 두어 파일을 열자마자 핵심 결과부터 보이게 함
    summary = {
        "timestamp": timestamp,
        "variant": args.variant,
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grad_clip_norm": args.grad_clip_norm,
            "calibrate_init": args.calibrate_init, "seed": args.seed,
            "patience": args.patience, "min_epochs": args.min_epochs,
            "class_weighted": args.class_weighted, "class_weights": class_weight_values,
            "sampler_weighted": args.sampler_weighted, "checkpoint_metric": args.checkpoint_metric,
        },
        # [2026-07-20 추가] 조기 종료 여부 명시 - actual_epochs < epochs면 patience에 걸려
        # 중단된 것(과적합 신호로 판단, best checkpoint는 항상 정상적으로 선택됨)
        "actual_epochs": actual_epochs,
        "early_stopped": actual_epochs < args.epochs,
        "best_epoch": test_result.get("best_epoch"),
        "best_val_acc": round(test_result.get("best_val_acc", 0.0), 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_precision_macro": round(test_result.get("precision_macro", float("nan")), 4),
        "test_recall_macro": round(test_result.get("recall_macro", float("nan")), 4),
        "test_f1_macro": round(test_result.get("f1_macro", float("nan")), 4),
        "elapsed_sec": round(elapsed, 1),
        "vs_paper_study2_diff_pct": round(diff * 100, 2) if args.variant in paper_acc_study2 else None,
    }

    history_rounded = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in epoch_row.items()}
        for epoch_row in history
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "test_result": test_result,
            "paper_vs_model_comparison": [
                {"항목": label, "실험값": model_v, "논문값": paper_v, "비고": note}
                for label, paper_v, model_v, note in comparison_rows
            ],
            "n_params": n_params,
            "history": history_rounded,
        }, f, ensure_ascii=False, indent=2)
    print(f"상세 결과 JSON 저장: {out_path}")

    # [2026-07-21 추가] 이제까지는 best-validation 가중치(best_state)를 평가에만 쓰고
    # 디스크에 저장하지 않아, 학습이 끝나면 가중치가 사라졌음(재실행마다 매번 새로 학습
    # 필요). 09_Service의 실제 추론/XAI 데모가 이 파일을 로드해서 쓸 수 있도록 저장.
    checkpoints_dir = os.path.join(_ROOT, "03_Model_Training", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"ablation_{args.variant}_{timestamp_fs}_acc{acc_tag}.pt")
    torch.save({
        "variant": args.variant,
        "model_state_dict": best_state,
        "num_classes": 3,
        "input_size": 56,
        "class_names": ["Control", "Prodromal", "PD"],
        "hyperparams": summary["hyperparams"],
        "test_accuracy": test_result["accuracy"],
        "test_precision_macro": test_result.get("precision_macro"),
        "test_recall_macro": test_result.get("recall_macro"),
        "test_f1_macro": test_result.get("f1_macro"),
        "best_epoch": best_epoch,
        "timestamp": timestamp,
    }, ckpt_path)
    print(f"체크포인트 저장 (09_Service 추론용): {ckpt_path}")


if __name__ == "__main__":
    main()