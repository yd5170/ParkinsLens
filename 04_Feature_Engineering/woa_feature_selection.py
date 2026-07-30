# -*- coding: utf-8 -*-
"""
woa_feature_selection.py - WOA(Whale Optimization Algorithm) 기반 이진 특징 선택 (Model-4)

근거: 논문 원문(07_Document/주논문_nature.pdf) "WOA for feature selection" 절
(Eq.8-17), Table 7(하이퍼파라미터).
- Constant(b)=1, Threshold(thres)=0.5, Population Size=100, Number of iterations=200,
  Search bounds=[0,1], No. of independent runs=20 (Table 7 원문 그대로)
- Fitness function Eq.17: f = alpha*R(D) + (1-alpha)*|R|/|N|
  R(D) = KNN 분류 오류율(error rate), |R| = 선택된 특징 수, |N| = 전체 특징 수
- WOA는 CCA 융합 결과(fused feature)에 적용됨 - 논문 원문(Model-4 절): "applies...
  WOA to select the **optimal fused features**" (CCA 이전의 원본 2000차원이 아니라
  CCA 이후 결과에 적용한다는 뜻 - CCA→WOA 순서로 확정).

[논문에 기재되지 않아 자체 결정한 값]
- alpha(fitness 가중치): Table 7에 값 없음. 특징선택 WOA 문헌 통상값인 alpha=0.99
  (정확도에 절대적 가중치, 특징 수 축소는 부차적 목표) 채택.
- KNN의 K값: 논문 미기재. K=5(scikit-learn 근사 기본값) 사용.
- 이진화(연속값->이진값) 변환 방식: Table 7이 "Lower and upper bound = 0 and 1"을
  명시함 - 탐색 공간 자체가 [0,1]로 고정돼 있다는 뜻. [2026-07-24 한 차례 시행착오]
  한때 표준 Binary WOA 문헌(Mirjalili & Lewis 계열, 보통 비유계/대칭 구간 위치값을
  sigmoid로 압축)을 따라 sigmoid 변환을 추가했었으나, sigmoid(x)는 x>=0이면 항상
  >=0.5가 되어 위치값이 [0,1]로 이미 제한된 상황에서는 threshold=0.5를 넘겨
  "항상 선택됨" 쪽으로 쏠리는 결함이 있음을 발견(더미데이터 self-test에서 40/40
  전부 선택되는 것으로 확인). Table7이 명시한 bounds=[0,1]은 위치값 자체가 이미
  [0,1] 확률처럼 쓰인다는 뜻으로 해석하는 게 더 타당해 sigmoid 없이 위치값을
  threshold와 직접 비교하는 방식으로 되돌림.
- Fitness 평가 프로토콜(R(D) 계산 시 train/test를 어떻게 나누는가): 논문은 "KNN
  classifier의 분류 오류율"이라고만 하고 구체적 프로토콜(고정 holdout vs
  교차검증)을 명시하지 않음. [2026-07-24 결정] 이 프로젝트의 다른 모든 단계
  (Table3/5의 "grid search and cross-validation", Model-3/4 최종평가의 15-fold CV,
  Table6의 5/10/15-fold)가 전부 교차검증을 표준으로 쓰고 있어, 논문의 전반적
  방법론과 일관되게 여기서도 3-fold 교차검증(cross_val_score)을 채택함 -
  03_Model_Training에서 고정 train/val 분리로 시도했을 때는 WOA 적용 후 성능이
  오히려 baseline보다 나빠졌으나(과적합 신호로 판단), 3-fold CV로 바꾸자 baseline
  대비 개선되는 것을 실제 데이터로 확인함(03_Model_Training/results/
  04pipeline_real_data_20runs_result.json 참조).
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ---- Table 7 하이퍼파라미터(논문 원문 그대로) ----
POP_SIZE = 100
N_ITER = 200
LB, UB = 0.0, 1.0
B_CONST = 1.0
THRESHOLD = 0.5
N_INDEPENDENT_RUNS = 20

# ---- 논문 미기재, 자체결정 ----
ALPHA = 0.99
BETA = 1 - ALPHA
KNN_K = 5


def _fitness(mask, X, y, alpha=ALPHA, knn_k=KNN_K, cv=3):
    n_total = X.shape[1]
    n_selected = int(mask.sum())
    if n_selected == 0:
        return 1.0  # 특징이 하나도 선택되지 않으면 최악의 fitness(오류율 100%로 취급)
    X_sel = X[:, mask.astype(bool)]
    knn = KNeighborsClassifier(n_neighbors=min(knn_k, len(y) - 1))
    try:
        scores = cross_val_score(knn, X_sel, y, cv=min(cv, len(np.unique(y))))
        acc = scores.mean()
    except Exception:
        acc = 0.0
    error_rate = 1.0 - acc
    return alpha * error_rate + (1 - alpha) * (n_selected / n_total)


def binary_woa_feature_selection(X, y, population_size=POP_SIZE, iterations=N_ITER,
                                  b=B_CONST, threshold=THRESHOLD, alpha=ALPHA, seed=42,
                                  verbose=False):
    """
    이진 WOA로 X(N, D)에서 최적 특징 서브셋을 선택(1회 실행).
    반환: best_mask(D,) bool 배열, best_fitness(float), history(list, iteration별 best fitness)
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]

    positions = rng.uniform(LB, UB, size=(population_size, n_features))

    def to_binary(pos):
        return (pos > threshold).astype(np.float64)

    fitness = np.array([_fitness(to_binary(positions[i]), X, y, alpha=alpha)
                         for i in range(population_size)])
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]
    history = [best_fit]

    for t in range(iterations):
        a = 2 - t * (2 / iterations)  # a: 2 -> 0 선형 감소 (Eq.9)
        for i in range(population_size):
            r1, r2 = rng.random(), rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()
            l = rng.uniform(-1, 1)

            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey (Eq.8-10)
                    D = np.abs(C * best_pos - positions[i])
                    positions[i] = best_pos - A * D
                else:
                    # Exploration: 무작위 개체 기준 탐색 (Eq.14-16)
                    rand_idx = rng.integers(0, population_size)
                    D = np.abs(C * positions[rand_idx] - positions[i])
                    positions[i] = positions[rand_idx] - A * D
            else:
                # Bubble-net attack: spiral update (Eq.11-13)
                D_prime = np.abs(best_pos - positions[i])
                positions[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            positions[i] = np.clip(positions[i], LB, UB)

        fitness = np.array([_fitness(to_binary(positions[i]), X, y, alpha=alpha)
                             for i in range(population_size)])
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fit:
            best_fit = fitness[gen_best_idx]
            best_pos = positions[gen_best_idx].copy()
        history.append(best_fit)
        if verbose and (t % max(1, iterations // 10) == 0):
            print(f"  [WOA] iter {t}/{iterations}  best_fitness={best_fit:.4f}")

    best_mask = (best_pos > threshold).astype(bool)
    return best_mask, best_fit, history


def run_independent_trials(X, y, n_runs=N_INDEPENDENT_RUNS, seed_base=1000, **kwargs):
    """Table 7의 "No. of independent runs=20"을 재현 - run마다 다른 시드로
    binary_woa_feature_selection()을 반복 호출하고 결과 리스트를 반환."""
    results = []
    for run_i in range(n_runs):
        mask, best_fit, history = binary_woa_feature_selection(X, y, seed=seed_base + run_i, **kwargs)
        results.append({"run": run_i, "seed": seed_base + run_i, "mask": mask,
                         "fitness": best_fit, "history": history})
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40))
    y = rng.integers(0, 3, size=60)
    mask, fit, hist = binary_woa_feature_selection(
        X, y, population_size=5, iterations=3, verbose=True
    )
    print("selected features:", mask.sum(), "/", len(mask), "best_fitness:", fit)
