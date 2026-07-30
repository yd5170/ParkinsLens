# -*- coding: utf-8 -*-
"""
app.py - 브레인텐서(BrainTensor) 서비스 화면 프로토타입 (Streamlit).

NeuroLens 목업 2장(① MRI 분석 대시보드, ② XAI 분석 리포트)을 참고해 같은 정보
구성으로 만들되, 확률/히트맵은 03_Model_Training/checkpoints의 실제 학습된
모델로부터 계산한 값을 그대로 사용한다(하드코딩 아님).

[중요 - 반드시 인지하고 사용할 것]
이 프로젝트는 논문 재현 연구용 프로토타입이며, 지금까지 학습된 체크포인트의
test accuracy는 30~52%대(3-클래스 랜덤 33%에 근접 ~ 다소 상회하는 수준)로,
논문이 보고한 82~90%에 크게 못 미친다(03_Model_Training/results/ 참조). 아래
화면은 "실제 진단 도구"가 아니라 "XAI 파이프라인이 실제로 동작한다"는 것을
보여주는 연구 프로토타입이다 - 이 disclaimer 문구를 임의로 지우지 말 것.

[실행 방법]
  1) (최초 1회, 로컬 GPU 환경) 03_Model_Training/train_ablation.py를 실행해
     checkpoints/*.pt 를 최소 1개 만든다.
     예) python ../03_Model_Training/train_ablation.py --variant variant1
  2) pip install streamlit nibabel   (다른 패키지는 프로젝트에 이미 있음)
  3) streamlit run app.py
"""
import datetime
import os
import sys

import numpy as np
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import inference as inf

st.set_page_config(page_title="BrainTensor", layout="wide")

CLASS_COLORS = {"Control": "#3b82f6", "Prodromal": "#f59e0b", "PD": "#ef4444"}
CLASS_LABELS_KR = {"Control": "정상", "Prodromal": "전구기", "PD": "파킨슨 의심"}


# ---------------------------------------------------------------- 시각화 유틸
def _jet_like(gray):
    """matplotlib 없이 blue->cyan->yellow->red 컬러맵을 근사 적용 (0~1 입력)."""
    gray = np.clip(gray, 0.0, 1.0)
    stops_t = np.array([0.0, 0.33, 0.66, 1.0])
    stops_c = np.array([
        [30, 60, 220],
        [0, 200, 200],
        [255, 220, 0],
        [220, 30, 30],
    ], dtype=np.float32)
    r = np.interp(gray, stops_t, stops_c[:, 0])
    g = np.interp(gray, stops_t, stops_c[:, 1])
    b = np.interp(gray, stops_t, stops_c[:, 2])
    return np.stack([r, g, b], axis=-1)


def _normalize01(a):
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        return (a - lo) / (hi - lo)
    return np.zeros_like(a)


def _upscale(img, size):
    """(h,w) 또는 (h,w,3) 배열을 size x size(채널은 그대로)로 확대."""
    from scipy.ndimage import zoom
    h, w = img.shape[:2]
    factors = [size / h, size / w] + ([1] if img.ndim == 3 else [])
    return zoom(img, factors, order=1)


# 축 인덱스 -> 표시 라벨. 볼륨이 등방(56^3)으로 리샘플된 뒤라 해부학적 방향이
# 엄밀히 검증되진 않았음 - 목업과 동일한 3분할(Axial/Coronal/Sagittal) 레이아웃을
# 보여주기 위한 근사 매핑이라는 점을 명시.
AXIS_LABELS = {0: "Axial(근사)", 1: "Coronal(근사)", 2: "Sagittal(근사)"}


def slice_overlay_rgb(volume, cam, axis, frac, size=260, cam_alpha_max=0.6):
    idx = int(np.clip(round(volume.shape[axis] * frac), 0, volume.shape[axis] - 1))
    base = np.take(volume, idx, axis=axis)
    heat = np.take(cam, idx, axis=axis)

    base_n = _normalize01(base)
    heat_n = np.clip(heat, 0.0, 1.0)

    base_rgb = np.stack([base_n] * 3, axis=-1) * 255.0
    heat_rgb = _jet_like(heat_n)

    alpha_map = (heat_n * cam_alpha_max)[..., None]
    composite = base_rgb * (1 - alpha_map) + heat_rgb * alpha_map
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    composite = _upscale(composite, size).astype(np.uint8)
    base_only = _upscale(np.stack([base_n] * 3, axis=-1) * 255.0, size).astype(np.uint8)
    return base_only, composite, idx


def probability_bars(probs, highlight=True):
    for cls in ["Control", "Prodromal", "PD"]:
        p = probs[cls]
        color = CLASS_COLORS[cls]
        label = CLASS_LABELS_KR[cls]
        st.markdown(
            f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                <span>{label} ({cls})</span><span><b>{p*100:.1f}%</b></span>
              </div>
              <div style="background:#e5e7eb;border-radius:4px;height:10px;overflow:hidden;">
                <div style="width:{max(p*100,1):.1f}%;background:{color};height:100%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------- 사이드바
st.sidebar.title("BrainTensor")
st.sidebar.caption("PPMI T2 MRI 기반 파킨슨병 XAI 분석 프로토타입")

checkpoints = inf.list_checkpoints()
if not checkpoints:
    st.sidebar.error(
        "checkpoints/*.pt 가 없습니다.\n\n"
        "03_Model_Training/train_ablation.py를 로컬 GPU 환경에서 먼저 실행해 "
        "체크포인트를 생성하세요."
    )
    st.stop()

ckpt_file = st.sidebar.selectbox("모델 체크포인트", checkpoints)

try:
    samples = inf.load_sample_list()
except FileNotFoundError as e:
    st.sidebar.error(f"데이터 경로를 찾을 수 없습니다: {e}")
    st.stop()

if not samples:
    st.sidebar.error("전처리된 볼륨을 가진 샘플이 없습니다. 01_Preprocessing 출력 경로를 확인하세요.")
    st.stop()

sample_options = {
    f"{s['sample_id']}  ({s['Group']}, {s['Sex']}/{s['Age']}세)": s for s in samples
}
sample_label = st.sidebar.selectbox("환자(샘플) 선택", list(sample_options.keys()))
selected_sample = sample_options[sample_label]

run_clicked = st.sidebar.button("분석 실행", type="primary", use_container_width=True)

page = st.sidebar.radio("화면", ["MRI 분석 대시보드", "XAI 분석 리포트"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ 연구용 프로토타입입니다. 현재 체크포인트의 test accuracy는 논문 목표치(82~90%)에 "
    "크게 못 미치는 수준으로, 임상적 진단 근거로 사용할 수 없습니다."
)

# ------------------------------------------------------------------- 추론 실행
state_key = "bt_result"
if run_clicked or state_key not in st.session_state:
    with st.spinner("모델 로드 및 Grad-CAM 계산 중..."):
        model, ckpt_meta = inf.load_model(ckpt_file)
        volume = inf.load_volume(selected_sample["sample_id"])
        result = inf.predict_with_cam(model, volume)
        st.session_state[state_key] = {
            "result": result,
            "volume": volume,
            "ckpt_meta": ckpt_meta,
            "sample": selected_sample,
            "ckpt_file": ckpt_file,
        }

data = st.session_state[state_key]
result, volume = data["result"], data["volume"]
ckpt_meta, sample = data["ckpt_meta"], data["sample"]


# ============================================================== 화면 1: 대시보드
def render_dashboard():
    st.markdown("## MRI 분석 대시보드")
    st.caption("Brain MRI 뇌 3D-MRI 병변 시각화 (M3d-CAM 근사 · Grad-CAM 기반)")

    col_main, col_side = st.columns([2.2, 1])

    with col_main:
        axis = st.slider("슬라이스 축", 0, 2, 0, format="%d", help="0=Axial(근사) / 1=Coronal(근사) / 2=Sagittal(근사)")
        frac = st.slider("슬라이스 위치", 0.0, 1.0, 0.5, step=0.02)
        base_img, overlay_img, idx = slice_overlay_rgb(volume, result["cam"], axis, frac)

        c1, c2 = st.columns(2)
        with c1:
            st.image(base_img, caption=f"원본 MRI (T2) - {AXIS_LABELS[axis]} slice {idx}", use_container_width=True)
        with c2:
            st.image(overlay_img, caption="AI 분석 결과 (Grad-CAM Heatmap)", use_container_width=True)

    with col_side:
        st.markdown("#### 환자 정보")
        st.write(f"**ID**: {sample['sample_id']}")
        st.write(f"**성별/연령**: {sample['Sex']} / {sample['Age']}세")
        st.write(f"**Scan Date**: {sample.get('Acq Date', '-')}")

        st.markdown("#### BrainTensor 판독 요약")
        probability_bars(result["probs"])

        st.markdown("#### 판단 근거")
        st.caption(
            f"모델: {ckpt_meta['variant']} ({data['ckpt_file']}) · "
            f"체크포인트 test accuracy: {ckpt_meta['test_accuracy']*100:.1f}%"
        )
        top_p = max(result["probs"].values())
        if top_p >= 0.7:
            st.warning(f"{CLASS_LABELS_KR[result['pred_label']]} 확률이 {top_p*100:.1f}%로 높게 분석되었습니다. 관련 전문의의 심층 검토를 권장합니다.")
        else:
            st.info(f"모델 예측: {CLASS_LABELS_KR[result['pred_label']]} ({top_p*100:.1f}%) - 클래스 간 확률 차이가 크지 않습니다.")


# ============================================================ 화면 2: XAI 리포트
def render_report():
    st.markdown("## XAI 분석 리포트")
    st.caption("AI 모델(3D-CNN Ablation Series) 기반의 분석 결과이며, 참고용입니다. 최종 판독 및 진단은 담당 전문의의 임상 판단을 요구합니다.")

    top = st.columns(2)
    with top[0]:
        st.markdown("##### Patient Info")
        st.write(f"환자 ID : {sample['sample_id']}")
        st.write(f"검사일 : {sample.get('Acq Date', '-')}")
        st.write(f"검사 유형 : {sample.get('Description', 'T2 MRI')}")
        st.write("판독 상태 : **자동 생성 완료**")
    with top[1]:
        st.markdown("##### AI 보조 판독 요약 (AI Model Info & Disclaimer)")
        st.write(
            f"본 리포트는 AI 모델({ckpt_meta['variant']}, Grad-CAM 기반) 분석 결과이며 "
            "참고용입니다. 최종 판독 및 진단은 담당 전문의의 임상 판단을 요구합니다."
        )
        st.write(f"AI 보조 시스템: BrainTensor v0.1 (프로토타입)")

    st.markdown("---")
    st.markdown("##### XAI 시각화 (Grad-CAM)")
    axis = 0
    base_img, overlay_img, idx = slice_overlay_rgb(volume, result["cam"], axis, 0.5)
    c1, c2 = st.columns(2)
    with c1:
        st.image(base_img, caption="원본 MRI (T2)", use_container_width=True)
    with c2:
        st.image(overlay_img, caption="AI 분석 결과 (Grad-CAM Heatmap)", use_container_width=True)

    st.markdown("---")
    st.markdown("##### AI 진단 확률 요약")
    probability_bars(result["probs"])

    st.markdown("---")
    st.markdown("##### 핵심 판독 요약")
    st.write(
        f"- 모델 예측 클래스: **{CLASS_LABELS_KR[result['pred_label']]} ({result['pred_label']})**, "
        f"확률 {result['probs'][result['pred_label']]*100:.1f}%"
    )
    st.write(
        f"- 참고: 이 체크포인트의 held-out test accuracy는 {ckpt_meta['test_accuracy']*100:.1f}%로, "
        "동일 데이터셋에서 무작위 추정(3-클래스, 약 33%) 대비 성능이 제한적입니다. "
        "임상 결정에 사용하지 마십시오."
    )

    st.caption(f"생성일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")


if page == "MRI 분석 대시보드":
    render_dashboard()
else:
    render_report()
