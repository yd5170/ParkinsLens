import os
import sys
import numpy as np
import torch

_ROOT = r"D:\new tensor"
sys.path.insert(0, os.path.join(_ROOT, "02_Model_Definition"))
sys.path.insert(0, os.path.join(_ROOT, "03_Model_Training"))

from ablation_models import CNN3D_Variant3
from final_resnet import ResNet3D
from train_ablation import build_loaders

# [2026-07-26 갱신] 이전 체크포인트(20260721_225937)는 2026-07-24 구조 변경 전 버전이라
# 현재 CNN3D_Variant3(37,077,123 파라미터)와 shape이 안 맞아 로드 자체가 실패함.
# class-weighted loss + WeightedRandomSampler + val_acc 체크포인트 기준(팀원 J와 동일)
# 조합으로 재학습해 클래스 붕괴를 해결한 최신 체크포인트로 교체
# (Test Acc=50.00%, Control/Prodromal/PD recall=65%/38%/43%, 3개 클래스 전부 recall>0%).
CNN_CKPT = os.path.join(_ROOT, "03_Model_Training", "checkpoints", "ablation_variant3_20260726_035445_acc50.0.pt")
RESNET_CKPT = os.path.join(_ROOT, "03_Model_Training", "checkpoints", "resnet3d_20260722_170643_acc54.3.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn():
    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    m = CNN3D_Variant3(num_classes=3).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt["hyperparams"]


def load_resnet():
    ckpt = torch.load(RESNET_CKPT, map_location="cpu", weights_only=False)
    m = ResNet3D(num_classes=3).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt["hyperparams"]


def algorithm1_feature_concatenation(x1, x2, w1=1.0, w2=1.0):
    """논문 Algorithm 1(Feature Concatenation) 그대로 구현.
    x1, x2: (N, D) numpy 배열(정규화 전). w1, w2: 스칼라 가중치(논문 미기재 - 자체결정
    동일가중치 1.0). 출력 x3: (N, D) - x1과 같은 길이(concat 아님, 원소별 선택).

    [2026-07-22 자체결정 사항]
    - Normalize: 논문이 "Normalize(x1 and x2)"라고만 하고 방법 미기재 - 샘플(행)별
      z-score 표준화 채택(평균0, 표준편차1), 서로 다른 스케일의 두 특징을 비교
      가능하게 만드는 가장 표준적인 선택.
    - used_features(중복 방지): 원문 그대로 "값"을 기준으로 추적. ReLU 계열 특징은
      0.0이 매우 흔해 두 번째 이후 0.0 후보는 전부 fallback으로 빠짐 - 논문
      pseudo-code 자체의 한계로 판단, 그대로 구현.
    - fallback_logic(): 논문에 정의 없음(단순히 "fallback_logic()"이라고만 표기) -
      두 후보 다 이미 used_features에 있으면 가중치 무시하고 그냥 더 큰 값을
      사용하는 것으로 자체결정(출력이 항상 정의되도록 하는 가장 단순한 방법).
    """
    n, d = x1.shape
    # 샘플별 z-score 정규화
    def znorm(a):
        mu = a.mean(axis=1, keepdims=True)
        sd = a.std(axis=1, keepdims=True) + 1e-8
        return (a - mu) / sd
    x1n, x2n = znorm(x1), znorm(x2)

    x3 = np.zeros_like(x1)
    for i in range(n):
        used = set()
        for j in range(d):
            wv1 = w1 * x1n[i, j]
            wv2 = w2 * x2n[i, j]
            v1, v2 = x1[i, j], x2[i, j]
            if wv1 > wv2 and v1 not in used:
                x3[i, j] = v1
                used.add(v1)
            elif wv2 >= wv1 and v2 not in used:
                x3[i, j] = v2
                used.add(v2)
            else:
                x3[i, j] = max(v1, v2)  # fallback: 자체결정
                used.add(x3[i, j])
    return x3


def extract_all(cnn, resnet, loader):
    """[2026-07-24 정리] CNN/ResNet 둘 다 각 모델 클래스 자신의
    forward(x, return_features=True)를 그대로 호출 - 이전엔 CNN은 별도 함수로
    forward를 통째로 재구현하고(모델에 return_features 옵션이 없었음), ResNet은
    이미 모델에 있는 기능을 쓰지 않고 또 별도로 재구현해서 두 갈래로 나뉘어 있었음.
    ablation_models.CNN3D_Variant3에 return_features를 추가해서(02_Model_Definition
    참조) 이제 두 모델 다 같은 방식(모델 자신의 forward)으로 특징을 뽑음 - 중복 제거."""
    fc1_list, fc2_list, fv4_list, y_list, sid_list = [], [], [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            _, cnn_feats = cnn(x, return_features=True)
            _, resnet_feats = resnet(x, return_features=True)
            fc1_list.append(cnn_feats["fc1"].cpu().numpy())
            fc2_list.append(cnn_feats["fc2"].cpu().numpy())
            fv4_list.append(resnet_feats["fc4"].cpu().numpy())
            y_list.append(y.numpy())
            sid_list.extend(sid)
    fc1_all = np.concatenate(fc1_list)
    fc2_all = np.concatenate(fc2_list)
    fv3_all = algorithm1_feature_concatenation(fc1_all, fc2_all)
    return fv3_all, np.concatenate(fv4_list), np.concatenate(y_list), sid_list


if __name__ == "__main__":
    cnn, cnn_hp = load_cnn()
    resnet, resnet_hp = load_resnet()
    print("CNN(Variant3) seed:", cnn_hp["seed"], "batch:", cnn_hp["batch_size"])
    print("ResNet seed:", resnet_hp["seed"], "batch:", resnet_hp["batch_size"])

    csv_path = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
    image_dir = os.path.join(_ROOT, "01_Preprocessing", "전처리_ref21order_v1")
    # 두 체크포인트 다 seed42라 동일 split 사용 - batch_size는 특징 추출용이니 임의(32)로 통일
    train_loader, val_loader, test_loader = build_loaders(csv_path, image_dir, 32, 42)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        fv3, fv4, y, sid = extract_all(cnn, resnet, loader)
        print(split_name, "FV-3(CNN):", fv3.shape, "FV-4(ResNet):", fv4.shape, "y:", y.shape)
        np.savez(os.path.join(out_dir, f"features_{split_name}.npz"), fv3=fv3, fv4=fv4, y=y, sid=np.array(sid))
    print("done")
