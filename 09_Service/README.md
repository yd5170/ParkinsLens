# 09_Service — BrainTensor 서비스 화면 프로토타입

NeuroLens 목업 2장(MRI 분석 대시보드 / XAI 분석 리포트)을 참고해 만든 Streamlit
데모입니다. 화면에 나오는 확률·히트맵은 `03_Model_Training/checkpoints`에 저장된
실제 학습 체크포인트로부터 계산한 값이며, 하드코딩된 값이 아닙니다.

## ⚠️ 먼저 알아야 할 것

- 지금까지 나온 최고 test accuracy는 52.2%대(`03_Model_Training/results/` 참조)로,
  논문이 보고한 82~90%에 크게 못 미칩니다. 클래스 붕괴(recall 0) 문제도 완전히
  해결되지 않았습니다. 이 화면은 **"XAI 파이프라인이 실제로 동작한다"는 것을
  보여주는 연구 프로토타입**이지, 신뢰할 수 있는 진단 도구가 아닙니다. 화면의
  disclaimer 문구는 지우지 마세요.
- 이 앱은 임의의 새 원본 MRI 업로드를 지원하지 않습니다. `01_Preprocessing`이
  이미 전처리(정합·두개골 제거·N4·리사이즈)를 마친 303명 데이터셋에서 환자(샘플)를
  골라 추론하는 방식입니다. 새 원본 파일을 지원하려면 `inference.py`에
  `01_Preprocessing` 파이프라인 호출 단계를 추가해야 합니다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `gradcam3d.py` | `ablation_models.py`의 4개 모델(Base/Variant1/Variant1_Untuned/Variant2) 각각에 맞춰 마지막 conv 레이어를 자동으로 골라 3D Grad-CAM을 계산 |
| `inference.py` | 체크포인트 로드, 샘플 목록/볼륨 로드, `predict_with_cam()`으로 확률+히트맵 반환 |
| `app.py` | Streamlit 앱 (사이드바에서 체크포인트/환자 선택 → 대시보드/리포트 화면 전환) |

## 실행 방법 (로컬 GPU 환경)

이 저장소의 개발 샌드박스는 torch DLL 문제로 실행 검증을 하지 못했습니다.
`train_ablation.py`와 동일하게 로컬 GPU 머신에서 실행하세요.

```bash
# 1) 체크포인트가 하나도 없다면 먼저 학습 (예: variant1, 약 15~20분)
cd 03_Model_Training
python train_ablation.py --variant variant1

# 2) 앱 의존성 설치 (nibabel/torch/scipy는 이미 설치돼 있다고 가정)
pip install streamlit

# 3) 앱 실행
cd ../09_Service
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 출력되는 `http://localhost:8501`로
접속하세요. 사이드바에서 체크포인트와 환자(샘플)를 고르고 "분석 실행"을 누르면
실제 모델 추론 + Grad-CAM 결과가 두 화면(대시보드/리포트)에 반영됩니다.

## 체크포인트가 왜 없었나

`train_ablation.py`는 이번 변경 전까지 best-validation epoch의 가중치를
메모리에만 들고 있다가 평가에 쓰고 버렸습니다(`results/*.json`에는 지표만 저장).
이번에 학습 스크립트 마지막에 `03_Model_Training/checkpoints/*.pt` 저장 코드를
추가했으니, 이 폴더에서 앞으로 실행하는 학습부터 체크포인트가 쌓입니다. 기존에
이미 만들어진 `results/*.json`들은 가중치가 없으므로 재학습이 필요합니다.
