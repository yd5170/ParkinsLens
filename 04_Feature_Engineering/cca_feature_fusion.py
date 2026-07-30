# -*- coding: utf-8 -*-
"""
cca_feature_fusion.py - CCA(Canonical Correlation Analysis) 특징 융합 (Model-3)

근거: 논문 원문(07_Document/주논문_nature.pdf) "Canonical correlation analysis" 절,
Eq.(3)-(7).

[2026-07-24 정정 - 원문 재확인 결과]
이 파일의 이전 버전(2026-07-03 작성)은 아래 두 가지가 논문과 달랐음을
03_Model_Training 세션에서 원문을 다시 대조해 확인 후 정정:

1. n_components에 자체적으로 100을 상한으로 걸어뒀었는데, 논문은 이런 상한을 전혀
   언급하지 않음. CCA 자체가 수학적으로 min(n_samples-1, p, q)를 넘는 성분을 만들
   수 없으므로("SAA/SBB의 고유값 분해가 이 개수를 넘는 nonzero eigenvalue를 만들
   수 없음"), 이 수학적 최댓값을 그대로 쓰는 것이 더 정확함 - 임의로 100을 캡핑할
   근거가 없었음.
2. 융합 방식이 concat(Eq.6)만 구현돼 있었는데, 논문 원문은 "The modified feature
   vectors are concatenated **or summed up**"(p.7)라고 명시적으로 두 방식을 모두
   언급함. 이번 버전은 두 방식(z_concat, z_sum) 모두 계산해서 반환 - 실제 데이터로
   비교한 결과 z_sum(Eq.7, 합산)이 z_concat(Eq.6, 이어붙이기)보다 더 나은 분류
   성능을 보였음(03_Model_Training/results 참조).

[FV-3(CNN 쪽 입력) 차원에 대한 참고 - 논문 원문 근거]
"Feature vector x~3 is with the dimensions FV1 x 1000 and FV2 x 1000"(p.6),
"A total of 2000 features were extracted from both the 3D Resnet and the 3D CNN
modules"(p.7) - 즉 CNN 쪽(FV-3, extract_features.py의 Algorithm1 결과)과 ResNet
쪽(FC-4)이 각각 1000차원씩이라 합쳐서 2000. 이 파일 자체는 입력 차원을 하드코딩
하지 않고 fv_a/fv_b의 실제 shape을 그대로 쓰므로, extract_features.py가 FV-3를
1000차원(Algorithm1)으로 만들어 넘겨주기만 하면 자동으로 논문과 일치함.
"""
import numpy as np
from sklearn.cross_decomposition import CCA

# [2026-07-26 재정정] 2026-07-24에 "논문이 상한을 언급하지 않는다"는 이유로 무상한
# (수학적 최댓값, min(n_samples-1,p,q))으로 바꿨는데, 실제로 train=212개 샘플에
# n_components=211이 나와 샘플 수와 성분 수가 거의 1:1이 되는 문제가 발견됨
# (2026-07-26 run_real_pipeline.py 실행 결과 분석) - 뒤쪽 성분 대부분이 진짜
# 상관관계가 아니라 훈련 데이터 노이즈를 과적합했을 가능성이 높고, 이 노이즈 섞인
# 422차원(z_concat) 공간에서 WOA가 특징을 고르다 보니 회차마다 선택 특징 수가
# 1개~364개로 요동치는 등 다운스트림 불안정성의 원인으로 추정됨(CCA+WOA 결과
# 30.22%가 Variant3 단독 50.00%보다 낮게 나온 것과 연관 가능성). 논문에 상한이
# 없다는 사실 자체는 맞지만, 이 프로젝트의 작은 표본 규모(212개)에서는 "논문
# 그대로"가 통계적으로 불안정한 결과를 낳는다고 판단해 과거(2026-07-24 이전)에
# 썼던 상한값(100)을 다시 적용.
MAX_N_COMPONENTS = 100


def cca_fuse(fv_a, fv_b, n_components=None, fitted_cca=None):
    """
    fv_a, fv_b: (N, D_a), (N, D_b) 특징 행렬 (예: FV-3(3D-CNN, 1000차원), FC-4(3D-ResNet, 1000차원))
    fitted_cca: 이미 학습된 CCA 객체가 있으면 재사용(테스트 세트 변환 시 사용,
                train 세트로 fit한 CCA를 그대로 재사용해야 데이터 누출을 방지함)

    반환: z_concat (N, 2*n_components) - Eq.6, z_sum (N, n_components) - Eq.7, cca 객체
          논문이 "concatenated or summed up"이라고 두 방식 다 언급하므로 둘 다 계산해서
          반환 - 실제 사용시 어느 쪽이 더 나은지는 데이터로 비교해서 결정할 것.
    """
    n_samples = fv_a.shape[0]
    if n_components is None:
        # [2026-07-26 재정정] 수학적 최댓값(min(n-1,p,q))에 MAX_N_COMPONENTS 상한을
        # 다시 적용 - 파일 상단 changelog 참조(작은 표본에서 과적합 방지 목적).
        math_max = max(1, min(n_samples - 1, fv_a.shape[1], fv_b.shape[1]))
        n_components = min(math_max, MAX_N_COMPONENTS)

    if fitted_cca is not None:
        cca = fitted_cca
        z1, z2 = cca.transform(fv_a, fv_b)
    else:
        cca = CCA(n_components=n_components, max_iter=2000)
        z1, z2 = cca.fit_transform(fv_a, fv_b)

    z_concat = np.concatenate([z1, z2], axis=1)  # Eq.6
    z_sum = z1 + z2                               # Eq.7
    return z_concat, z_sum, cca


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 더미 데이터 self-test - 논문대로 FV-3=1000, FC-4=1000
    fv3 = rng.normal(size=(30, 1000))
    fc4 = rng.normal(size=(30, 1000))
    z_concat, z_sum, cca = cca_fuse(fv3, fc4)
    print("z_concat(Eq6) shape:", z_concat.shape, "z_sum(Eq7) shape:", z_sum.shape,
          "n_components:", cca.n_components)
