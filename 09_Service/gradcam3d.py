# -*- coding: utf-8 -*-
"""
gradcam3d.py - 02_Model_Definition/ablation_models.py 모델들을 위한 3D Grad-CAM.

NeuroLens 목업의 "M3d-CAM" 히트맵과 동일한 역할(마지막 conv 레이어의 활성값 x
그 레이어에 대한 예측 클래스의 gradient로 관심 영역을 시각화)을 하지만, M3d-CAM
패키지 자체를 쓰지 않고 표준 Grad-CAM(Selvaraju et al., 2017) 알고리즘을 3D로
직접 구현했다 - 이 프로젝트의 ablation 모델들(conv 레이어 개수/구조가 서로 다름,
Base/Variant1/Variant1_Untuned/Variant2/Variant3)에 맞춰 대상 레이어를 자동으로
고르기 위함.

[설계 결정]
- 대상 레이어(target layer) = 각 모델의 "마지막 conv" 모듈. ReLU 이전/이후 어느
  시점을 쓰느냐는 Grad-CAM 원 논문에서도 구현마다 다른데, 여기서는 conv 모듈
  자체의 출력(ReLU 적용 전)에 hook을 걸고 CAM 자체에 마지막에 ReLU를 적용하는
  표준 방식을 따른다.
- CAM 해상도는 대상 레이어의 공간 해상도(예: Base는 pooling 전이라 56^3, Variant2는
  블록2 conv 출력이라 28^3)를 그대로 따르고, 원본 볼륨 크기로 리사이즈하는 건
  호출하는 쪽(inference.py)의 책임으로 둔다.
"""
import torch


def get_target_layer(model):
    """모델 클래스명에 따라 Grad-CAM 대상(마지막 conv) 레이어를 반환한다."""
    cls_name = model.__class__.__name__
    if cls_name == "CNN3D_Base":
        return model.conv2
    if cls_name in ("CNN3D_Variant1", "CNN3D_Variant1_Untuned"):
        return model.conv3
    if cls_name == "CNN3D_Variant2":
        return model.conv5
    if cls_name == "CNN3D_Variant3":
        # [2026-07-26 추가] 최종 모델(Table3 Study2 확정 구성) 지원. 블록3의 마지막
        # conv(1024->512->256 중 conv7, 256채널) 출력이 pool3/bn3/global_max_pool을
        # 거쳐 분류로 이어짐 - 다른 variant와 동일하게 "마지막 conv" 기준.
        return model.conv7
    raise ValueError(f"Grad-CAM 대상 레이어가 정의되지 않은 모델: {cls_name}")


class GradCAM3D:
    """3D CNN용 Grad-CAM. 사용 후 반드시 close()로 hook을 해제할 것(재사용 시 누적 방지)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._grad_handle = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        """[2026-07-26 수정] 원래는 register_full_backward_hook을 target_layer에
        걸었는데, CNN3D_Variant3에서 conv7 바로 뒤에 self.relu(inplace=True)가
        적용돼 "Output 0 of BackwardHookFunctionBackward is a view and is being
        modified inplace" RuntimeError가 발생함(실측 확인, 09_Service/inference.py로
        Variant3 체크포인트 실제 추론 중 발견). 모델의 inplace ReLU는 학습 성능에
        영향을 주므로 건드리지 않고, 대신 모듈 단위 backward hook 대신 출력 텐서에
        직접 register_hook을 거는 방식으로 변경 - 이 inplace-view 충돌을 피하는
        PyTorch의 일반적인 우회법이며, 다른 variant(Base/V1/V2)에도 동일하게
        적용해도 계산 결과는 동일함(grad_output[0]과 텐서 자체 gradient가 이
        단일-출력 케이스에서 같은 값)."""
        self._activations = output
        if self._grad_handle is not None:
            self._grad_handle.remove()
        self._grad_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad.detach()

    def __call__(self, x, class_idx=None):
        """x: (1, 1, D, H, W) 텐서. class_idx가 None이면 예측 클래스 기준으로 계산.

        반환: (cam, probs, pred_idx)
          - cam: (d, h, w) numpy, 0~1 정규화된 3D 히트맵 (대상 레이어 해상도)
          - probs: (num_classes,) numpy, softmax 확률
          - pred_idx: 모델이 예측한 클래스 인덱스(int)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())
        target_idx = pred_idx if class_idx is None else class_idx

        score = logits[:, target_idx]
        score.backward()

        # 채널별로 (D,H,W) 축에 대해 gradient를 전역 평균 풀링 -> Grad-CAM 가중치
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self._activations.detach()).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0]

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = cam_np * 0.0

        return cam_np, probs.detach().cpu().numpy()[0], pred_idx

    def close(self):
        self._fwd_handle.remove()
        if self._grad_handle is not None:
            self._grad_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
