# -*- coding: utf-8 -*-
"""
profile_variant3_layers.py - CNN3D_Variant3(Model-1, ablation_models.py) 레이어별
forward 소요시간 프로파일링.

ablation_models.py의 forward() 로직 자체는 건드리지 않고, 이 스크립트에서 동일한
순서로 서브모듈을 하나씩 직접 호출하면서 각 레이어 전후에 torch.cuda.synchronize()를
넣어 GPU 연산이 실제로 끝난 시점을 기준으로 정확히 측정함(sync 없이 재면 CUDA는
비동기라 호출 자체는 즉시 반환되어 잘못된 값이 나옴).

측정 대상:
1. 실제 학습 데이터(Table3 Study2 batch=32)로 한 epoch만큼 배치를 돌며 레이어별
   forward 시간을 누적/평균.
2. backward+optimizer.step()은 레이어별로 쪼개지 않고(자동미분 특성상 정확한
   레이어별 분리가 어려움) 배치당 통짜로 측정.
3. 위 1~2는 sync를 계속 넣어서 측정하므로 계측 오버헤드가 포함됨 - 계측 없는
   "순정" 배치당 총 시간(forward+backward+step)도 별도로 몇 배치 추가 측정해서
   비교 기준으로 같이 제공.

[2026-07-23 실행 시점 참고] 이 시점에 Variant1 k-fold(GPU) 작업이 동시에 돌고 있어
GPU 자원을 나눠 씀 - 아래 측정치는 "단독 실행" 기준보다 다소 부풀려져 있을 수 있음
(사용자 확인 후 진행하기로 함).

출력: time.log(모든 배치의 상세 로그) + 콘솔 요약표.
"""
import os
import sys
import time

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _rel in ["02_Model_Definition", "03_Model_Training"]:
    _p = os.path.join(_ROOT, _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ablation_models import CNN3D_Variant3
from train_ablation import build_loaders

LOG_PATH = os.path.join(_THIS_DIR, "time.log")
N_TIMED_BATCHES = 7      # 배치=32, train 212개 -> 한 epoch 분량(7배치)
N_CLEAN_BATCHES = 5      # 계측 오버헤드 없는 "순정" 배치 시간 비교용


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync()
    return out, time.perf_counter() - t0


def forward_with_layer_timing(model, x):
    """CNN3D_Variant3.forward()와 완전히 동일한 순서/연산을 레이어 단위로 직접 호출."""
    timings = {}

    x, timings["conv1"] = timed_call(model.conv1, x)
    x, t_relu1 = timed_call(model.relu, x)
    x, timings["conv2"] = timed_call(model.conv2, x)
    x, t_relu2 = timed_call(model.relu, x)
    timings["relu(conv1,conv2)"] = t_relu1 + t_relu2
    x, timings["pool1"] = timed_call(model.pool1, x)
    x, timings["bn1"] = timed_call(model.bn1, x)

    x, timings["conv3"] = timed_call(model.conv3, x)
    x, t_relu3 = timed_call(model.relu, x)
    x, timings["conv4"] = timed_call(model.conv4, x)
    x, t_relu4 = timed_call(model.relu, x)
    x, timings["conv5"] = timed_call(model.conv5, x)
    x, t_relu5 = timed_call(model.relu, x)
    timings["relu(conv3,conv4,conv5)"] = t_relu3 + t_relu4 + t_relu5
    x, timings["pool2"] = timed_call(model.pool2, x)
    x, timings["bn2"] = timed_call(model.bn2, x)

    x, timings["conv6"] = timed_call(model.conv6, x)
    x, t_relu6 = timed_call(model.relu, x)
    x, timings["conv7"] = timed_call(model.conv7, x)
    x, t_relu7 = timed_call(model.relu, x)
    timings["relu(conv6,conv7)"] = t_relu6 + t_relu7
    x, timings["pool3"] = timed_call(model.pool3, x)
    x, timings["bn3"] = timed_call(model.bn3, x)

    x, timings["global_max_pool"] = timed_call(model.global_max_pool, x)
    x, timings["flatten"] = timed_call(model.flatten, x)

    fc1_feat, timings["fc1"] = timed_call(model.fc1, x)
    fc2_feat, timings["fc2"] = timed_call(model.fc2, x)

    sync()
    t0 = time.perf_counter()
    fv3 = torch.cat([fc1_feat, fc2_feat], dim=1)
    sync()
    timings["cat(fc1,fc2)"] = time.perf_counter() - t0

    logits, timings["classifier"] = timed_call(model.classifier, fv3)
    return logits, timings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
    image_dir = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")
    train_loader, val_loader, test_loader = build_loaders(csv_path, image_dir, batch_size=32, seed=42)

    model = CNN3D_Variant3(num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[모델] CNN3D_Variant3(Model-1), 파라미터 수={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()

    log_lines = []
    log_lines.append(f"CNN3D_Variant3 레이어별 시간 프로파일링 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Device={device}, batch_size=32, n_params={n_params:,}")
    log_lines.append(f"주의: Variant1 k-fold 작업과 GPU를 동시에 사용 중 - 단독실행 대비 부풀려질 수 있음")
    log_lines.append("=" * 70)

    # ---- 웜업(측정 제외) ----
    it = iter(train_loader)
    for _ in range(2):
        try:
            x, y, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, _ = next(it)
        x, y = x.to(device), y.to(device)
        logits, _ = forward_with_layer_timing(model, x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[웜업 완료] 2배치")

    # ---- 계측 배치(레이어별 forward + 배치당 통짜 backward/step) ----
    all_layer_timings = []
    total_forward_sum = 0.0
    total_backward_step_sum = 0.0
    it = iter(train_loader)
    for b in range(N_TIMED_BATCHES):
        try:
            x, y, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, _ = next(it)
        x, y = x.to(device), y.to(device)

        logits, timings = forward_with_layer_timing(model, x)
        batch_forward_total = sum(timings.values())
        total_forward_sum += batch_forward_total

        sync()
        t0 = time.perf_counter()
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync()
        backward_step_time = time.perf_counter() - t0
        total_backward_step_sum += backward_step_time

        all_layer_timings.append(timings)
        line = f"[배치 {b+1}/{N_TIMED_BATCHES}] forward_total={batch_forward_total*1000:.2f}ms backward+step={backward_step_time*1000:.2f}ms"
        print(line)
        log_lines.append(line)
        for lname, t in timings.items():
            sub = f"    {lname:<24s} {t*1000:8.3f} ms"
            log_lines.append(sub)

    # ---- 순정(계측 오버헤드 없는) 배치 시간 - 비교 기준 ----
    clean_times = []
    it = iter(train_loader)
    for b in range(N_CLEAN_BATCHES):
        try:
            x, y, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, _ = next(it)
        x, y = x.to(device), y.to(device)
        sync()
        t0 = time.perf_counter()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync()
        clean_times.append(time.perf_counter() - t0)
    clean_mean = sum(clean_times) / len(clean_times)

    # ---- 집계 ----
    layer_names = list(all_layer_timings[0].keys())
    layer_mean = {ln: sum(bt[ln] for bt in all_layer_timings) / len(all_layer_timings) for ln in layer_names}
    layer_total = {ln: sum(bt[ln] for bt in all_layer_timings) for ln in layer_names}
    forward_mean = total_forward_sum / N_TIMED_BATCHES
    backward_step_mean = total_backward_step_sum / N_TIMED_BATCHES
    instrumented_batch_mean = forward_mean + backward_step_mean

    n_train_samples = len(train_loader.dataset)
    epoch_estimate_instrumented = instrumented_batch_mean * len(train_loader)
    epoch_estimate_clean = clean_mean * len(train_loader)

    log_lines.append("=" * 70)
    log_lines.append("[레이어별 평균 forward 시간 (계측 배치 %d개 평균)]" % N_TIMED_BATCHES)
    for ln in layer_names:
        pct = layer_mean[ln] / forward_mean * 100 if forward_mean > 0 else 0
        log_lines.append(f"  {ln:<24s} 평균 {layer_mean[ln]*1000:8.3f} ms  ({pct:5.1f}%)  누적 {layer_total[ln]*1000:9.3f} ms")
    log_lines.append("-" * 70)
    log_lines.append(f"forward 합계(평균/배치)      : {forward_mean*1000:9.3f} ms")
    log_lines.append(f"backward+step(평균/배치)     : {backward_step_mean*1000:9.3f} ms")
    log_lines.append(f"계측 배치 총합(평균/배치)     : {instrumented_batch_mean*1000:9.3f} ms")
    log_lines.append(f"순정 배치 시간(평균, 계측없음) : {clean_mean*1000:9.3f} ms  ({N_CLEAN_BATCHES}배치 평균)")
    log_lines.append(f"계측 오버헤드                : {(instrumented_batch_mean-clean_mean)*1000:9.3f} ms/배치 ({(instrumented_batch_mean/clean_mean-1)*100:.1f}%)")
    log_lines.append("-" * 70)
    log_lines.append(f"1 epoch(배치 {len(train_loader)}개, train {n_train_samples}개) 예상 시간:")
    log_lines.append(f"  - 계측 기준 추정: {epoch_estimate_instrumented:.2f}초")
    log_lines.append(f"  - 순정 기준 추정(더 정확): {epoch_estimate_clean:.2f}초")
    log_lines.append(f"30 epoch 총 예상 시간(순정 기준): {epoch_estimate_clean*30/60:.1f}분 ({epoch_estimate_clean*30/3600:.2f}시간)")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n상세 로그 저장: {LOG_PATH}")

    # ---- 콘솔 요약표 ----
    print("\n" + "=" * 70)
    print(f"[요약] CNN3D_Variant3 레이어별 forward 시간 (계측 {N_TIMED_BATCHES}배치 평균, batch_size=32)")
    print("=" * 70)
    print(f"{'레이어':<24s}{'평균(ms)':>12s}{'비율(%)':>10s}")
    for ln in layer_names:
        pct = layer_mean[ln] / forward_mean * 100 if forward_mean > 0 else 0
        print(f"{ln:<24s}{layer_mean[ln]*1000:12.3f}{pct:10.1f}")
    print("-" * 70)
    print(f"{'forward 합계':<24s}{forward_mean*1000:12.3f}")
    print(f"{'backward+step':<24s}{backward_step_mean*1000:12.3f}")
    print(f"{'순정 배치 총시간':<24s}{clean_mean*1000:12.3f}")
    print(f"\n1 epoch 예상(순정 기준): {epoch_estimate_clean:.2f}초  |  30 epoch 예상: {epoch_estimate_clean*30/60:.1f}분")


if __name__ == "__main__":
    main()
