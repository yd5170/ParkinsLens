# 02_Model_Definition

주논문 Figure 2/3에 명시된 3D-CNN(Model-1), 3D-ResNet(Model-2) 아키텍처 재구현입니다.
학습 로직은 없고 `nn.Module` 클래스 정의만 있습니다 - 실제 학습은 `03_Model_Training`.

## 3D-CNN 파이프라인 단계 (2026-07-26 정리, 2026-07-27 최종 모델 파일 분리)

3D-CNN은 이 프로젝트에서 두 단계를 거쳐 최종 구조가 확정됐습니다. 파일명만 보고도
최종본이 뭔지 알 수 있도록 파일 자체를 분리했습니다.

```
1차 모델(Stage 1)              2차/최종 확정 모델(Stage 2)
cnn3d_figure2.py               ablation_models.py
CNN3D_Figure2                  (CNN3D_Base/Variant1/Variant1_Untuned/Variant2/CNN3D_Variant3
(Figure2 다이어그램               - Table3 Study1·2 ablation 실험 전체 기록)
 문자 그대로 재현)                        │
        │                      CNN3D_Variant3 = ablation 중 가장 성공적인 CNN
        │                                        │
        │                                        ▼
        │                      final_cnn.py               final_resnet.py
        │                      CNN3D(= CNN3D_Variant3 별칭)  ResNet3D
        │                      "최종 CNN을 담는 파일"          "최종 ResNet을 담는 파일"
        ▼                                        ▼                    ▼
train_cnn3d.py로 학습            train_ablation.py --variant variant3   train_resnet.py 로 학습
(참고용, 논문 비교 목표치 없음)     로 학습 (CNN 현재 확정 체크포인트:
                                checkpoints/ablation_variant3_20260726_035445_acc50.0.pt)
```

**번호 접두사(`01_`/`02_`) 대신 서술적인 파일명을 씁니다** — 어차피 파일명에 굳이
번호를 붙이지 않아도, `cnn3d_figure2.py`(1차) vs `final_cnn.py`/`final_resnet.py`(최종)라는
이름 자체가 역할을 드러냅니다. 참고로 Python은 숫자로 시작하는 모듈명을 `import`할 수
없다는 제약도 있습니다(`import 01_models`는 문법 오류).

**최종 모델을 CNN/ResNet 두 파일로 나눈 이유**: "final cnn과 final resnet이라고 나눠서
올려달라"는 요청 - 하나의 `final_models.py`에 둘 다 있던 걸 분리해서, 파일명만 보고도
어느 파일이 CNN이고 어느 파일이 ResNet인지 바로 알 수 있게 함.

## 구성

| 파일 | 내용 | 역할 |
|---|---|---|
| `cnn3d_figure2.py` | `CNN3D_Figure2` | 1차 모델 - Figure2 다이어그램을 직접 재현(채널 32→64→128→256→512→1024→512→256, FC-1/FC-2 각 1000차원 → concat 2000차원) |
| `ablation_models.py` | `CNN3D_Base`(8계층), `CNN3D_Variant1`(9계층), `CNN3D_Variant1_Untuned`(Study1 튜닝전 재현용), `CNN3D_Variant2`(17계층), `CNN3D_Variant3`(24계층) | Table 3 ablation 실험 전체 기록. **`CNN3D_Variant3`가 이 중 가장 성공적인 CNN 모델(Table3 Study2 그리드서치 확정 구성)** |
| **`final_cnn.py`** | **`CNN3D`**(= `CNN3D_Variant3`의 별칭) | **최종 CNN 모델을 담는 파일** |
| **`final_resnet.py`** | **`ResNet3D`**(15계층 = Residual Block 5개×3유닛, FC-4 1000차원, Dropout 포함) | **최종 ResNet 모델을 담는 파일** |

**클래스 본체는 각 파일에 하나씩만 존재**합니다 — `final_cnn.py`의 `CNN3D`는
`ablation_models.py`의 `CNN3D_Variant3`를 import해서 이름만 다시 노출한 별칭이지,
코드를 복제한 게 아닙니다(구조를 두 곳에 중복 정의하면 한쪽만 고치고 잊어버리는
사고가 나기 쉬워서 - 이 프로젝트에서 실제로 몇 번 겪은 문제입니다). `final_resnet.py`는
`ResNet3D`의 유일한 정의 위치입니다(재노출 아님, 이 파일이 곧 단일 소스).

## ⚠️ "FV-3"라는 이름이 두 군데서 다른 뜻으로 쓰입니다 - 헷갈리지 말 것

1. **CNN3D 계열(`CNN3D_Figure2`, `CNN3D_Variant3`/`final_cnn.CNN3D`) 내부 FV-3(=fv3)**:
   각 모델 자기 자신의 SoftMax 분류기에 들어가는 값. FC-1(1000)+FC-2(1000)을
   **단순 concat**해서 2000차원. 이건 모델 구조 자체의 일부이고 두 모델 다 동일한
   방식(둘 다 `forward(return_features=True)`로 확인 가능).
2. **`04_Feature_Engineering/extract_features.py`가 만드는 FV-3**: CCA 융합(Model-3)
   입력용으로 별도 추출하는 값. 논문 Algorithm 1(원소별 max 선택)을 그대로 구현해서
   **1000차원**(concat 아님). 최종 모델(`CNN3D_Variant3`) 체크포인트에서 FC-1/FC-2를
   다시 뽑아 Algorithm1로 합친 것이라, 1번의 "모델 자체 분류용 FV-3"와는 별개 계산입니다.

즉 "CNN3D 모델 자신이 쓰는 FV-3(2000, concat)"과 "CCA 파이프라인에 넘기는 FV-3
(1000, Algorithm1)"은 이름만 같고 다른 값입니다 - 2026-07-24에 이 구분이 명확해짐.

## 논문에 없어서 자체 결정한 값 (근거는 파일 상단 docstring 참고)

- Conv3D stride=1, padding=1
- `CNN3D_Figure2`의 Flatten 차원을 논문 명시값(12544)에 맞추기 위한 depth-collapse 방식
- `ablation_models.py` Base/Variant1~3의 conv 채널 폭(논문이 개수만 명시, 폭은 미기재)
- 3D-ResNet 채널 폭(64→128→256→512→512, 논문 Figure 3에 수치 없음)
- `final_resnet.py`의 Dropout(p=0.5, FC-4→classifier 사이) - Table5에 없는 항목,
  95M 파라미터 대비 train 212개뿐이라 실측 과적합 대응(2026-07-26)

자세한 논문-vs-실제 비교는 `03_Model_Training/DEVIATIONS.md` 1·2·9번 섹션 참고.

## 빠른 확인

```bash
python cnn3d_figure2.py
# CNN3D_Figure2(1차) logits: torch.Size([2, 3]) | fv3: torch.Size([2, 2000]) | flatten_dim: 12544

python final_cnn.py
# CNN3D(최종=Variant3) logits: torch.Size([2, 3]) | fv3: torch.Size([2, 2000]) | gmp_dim: 256

python final_resnet.py
# ResNet3D logits: torch.Size([2, 3]) | fc4: torch.Size([2, 1000])
```

## 최종 모델을 코드에서 쓰는 법

```python
from final_cnn import CNN3D         # = ablation_models.CNN3D_Variant3 (최종 CNN)
from final_resnet import ResNet3D   # Model-2, 최종 ResNet

# 1차 모델(Figure2, 참고/재현용)을 쓰고 싶을 때만:
from cnn3d_figure2 import CNN3D_Figure2

# ablation 실험 과정 전체(Base/V1/V2/V3)를 보고 싶을 때:
from ablation_models import CNN3D_Base, CNN3D_Variant1, CNN3D_Variant2, CNN3D_Variant3
```
