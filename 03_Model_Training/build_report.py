# -*- coding: utf-8 -*-
"""
build_own_report_v2.py - cnn_training_report_new_tensor_내결과.xlsx 갱신판

기존 build_own_report.py 대비 추가된 것:
1. ResNet3D(Model-2)을 모든 시트(Model Summary/Structure/Paper Architecture/
   Confusion Matrix/Run Settings/Assumptions/Epoch Metrics/Best Validation/
   Split Counts)에 CNN 4개 변형과 나란히 포함.
2. Epoch Metrics의 precision/recall/f1 칸 - 지금까지 전부 "없음"이었는데, 2026-07-23에
   train_ablation.py/train_resnet.py의 학습 루프(train_loop)가 epoch마다 macro
   precision/recall/f1을 기록하도록 바뀌었음. 이 스크립트는 결과 JSON에 해당 필드가
   있으면 채우고, 없으면(과거 형식 JSON) 여전히 "없음"으로 남김 - 재학습 없이도 새
   실행 결과가 나오면 자동으로 채워짐.
3. 신규 "K-Fold Results" 시트 - results/ 폴더에서 kfold_*.json(집계 결과)을 자동으로
   스캔해서 fold별 값 + 평균/표준편차를 채움. 아직 k-fold를 실행하지 않았으면 헤더만
   있는 빈 시트로 생성됨(나중에 k-fold를 실행하고 이 스크립트를 다시 돌리기만 하면
   자동으로 채워짐 - 수작업 편집 불필요).
4. weight_decay/k-fold Assumptions 문구를 Table5 확정값 기준으로 갱신
   ("자체결정" -> "논문 명시값 그대로 사용").
"""
import glob
import json
import os
import sys
import shutil
import openpyxl
from openpyxl.styles import Font, PatternFill

_ROOT = r"D:\new tensor"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "02_Model_Definition"))
sys.path.insert(0, os.path.join(_ROOT, "03_Model_Training"))
# [2026-07-26 추가] compute_confusion/compute_confusion.analyze_split이 05_Model_Evaluation로
# 옮겨진 뒤 이 경로가 빠져 있어 ModuleNotFoundError 발생 - 추가.
sys.path.insert(0, os.path.join(_ROOT, "05_Model_Evaluation"))

# [2026-07-26 갱신] "cnn_training_report (1).xlsx"는 Downloads에서 이미 정리돼 없어짐 -
# 같은 내용의 팀원 파일(cnn_training_report_J.xlsx)로 교체.
TEMPLATE_FILE = r"C:\Users\user\Downloads\cnn_training_report_J.xlsx"
OUT_FILE = r"C:\Users\user\Downloads\cnn_training_report_new_tensor_내결과.xlsx"

RESULTS_DIR = os.path.join(_ROOT, "03_Model_Training", "results")
CKPT_DIR = os.path.join(_ROOT, "03_Model_Training", "checkpoints")

RUN_FILES = {
    "base": "ablation_base_20260721_234341_acc52.2.json",
    "variant1": "ablation_variant1_20260721_205719_acc56.5.json",
    "variant2": "ablation_variant2_20260722_011833_acc26.1.json",
    # [2026-07-26 갱신] 이전 파일(20260721_225937)은 2026-07-24 구조 변경 전 버전이라
    # 지금 CNN3D_Variant3와 shape이 안 맞아 로드 자체가 실패함. class-weighted loss +
    # WeightedRandomSampler + val_acc 체크포인트 기준(팀원 J와 동일) 조합으로 재학습해
    # 클래스 붕괴를 해결한 최신 결과로 교체(Test Acc 50.00%, 3개 클래스 전부 recall>0%).
    "variant3": "ablation_variant3_20260726_035445_acc50.0.json",
    "cnn3d": "cnn3d_20260724_015432_acc45.7.json",
    "resnet3d": "resnet3d_20260722_170643_acc54.3.json",
}
CKPT_FILES = {
    "base": "ablation_base_20260721_234341_acc52.2.pt",
    "variant1": "ablation_variant1_20260721_205719_acc56.5.pt",
    "variant2": "ablation_variant2_20260722_011833_acc26.1.pt",
    "variant3": "ablation_variant3_20260726_035445_acc50.0.pt",
    "cnn3d": "cnn3d_20260724_015432_acc45.7.pt",
    "resnet3d": "resnet3d_20260722_170643_acc54.3.pt",
}
# [2026-07-26 갱신] "6개 모델 아니고 base/variant1/variant2는 지워달라"는 요청 -
# 리포트에는 최종 확정 모델(variant3)과 1차/참고 모델(cnn3d=Figure2), ResNet3D만
# 남기고 ablation 중간 단계(base/variant1/variant2)는 제외. 코드/RUN_FILES/CKPT_FILES
# 정의는 그대로 남겨둠(ablation_models.py 자체에는 이 클래스들이 여전히 존재하고,
# 나중에 다시 필요하면 ALL_VARIANTS에 도로 추가하면 됨).
ABLATION_VARIANTS = ["base", "variant1", "variant2", "variant3"]
ALL_VARIANTS = ["variant3", "cnn3d", "resnet3d"]
MODEL_TYPE = {
    "base": "ablation", "variant1": "ablation", "variant2": "ablation",
    # [2026-07-26 갱신, 2026-07-27 파일명 갱신] models.py 파일 분리(cnn3d_figure2.py=1차 /
    # final_cnn.py+final_resnet.py=최종) 작업으로 용어 정리됨 - variant3가 이제 명실상부한
    # 최종 모델(final_cnn.py의 CNN3D가 이 클래스의 별칭), cnn3d(Figure2)는 1차/참고 모델로 재분류.
    "variant3": "final_model", "cnn3d": "legacy_model", "resnet3d": "resnet",
}

# [2026-07-23 추가] 내부 코드 식별자(base/variant1/.../resnet3d)를 시트에 그대로 적으면
# ResNet이 CNN ablation 변형들(Base/Variant1~3)과 뭐가 다른지 한눈에 안 들어와서, 각
# 행이 실제로 담고 있는 내용(구조 계열/레이어 수/논문 근거)에 맞는 사람이 읽을 제목으로
# 별도 표시. 내부 딕셔너리 키(RUN_FILES 등)는 그대로 원래 코드명을 쓰고, 시트에 값을
# "쓸 때"만 이 매핑을 거침.
DISPLAY_NAME = {
    "base": "Base (CNN Ablation, 8-layer, Table3 Study2)",
    "variant1": "Variant 1 (CNN Ablation, 9-layer, Table3 Study2)",
    "variant2": "Variant 2 (CNN Ablation, 17-layer, Table3 Study2)",
    "variant3": "Variant 3 (CNN 최종 확정 모델, 24-layer, Table3 Study2 그리드서치 결과)",
    "cnn3d": "CNN3D_Figure2 (1차 모델, 24-layer, Figure2 다이어그램 직접 재현)",
    "resnet3d": "ResNet3D (Model-2 최종 확정 모델, 15-layer, Figure3/Table5)",
}

PAPER_ACC = {
    "base": (84.96, 87.34, 81.54, 85.32),
    "variant1": (88.25, 89.93, 84.28, 87.43),
    "variant2": (90.28, 91.83, 87.72, 90.81),
    "variant3": (93.41, 95.76, 88.98, 91.68),
    # [2026-07-24 추가] cnn3d(models.py CNN3D)는 Table3 Study2 그리드서치 대상이 아니라
    # 직접 대응하는 논문 수치가 없음. CNN3D_Variant3와 "같은 24-layer 최종모델"을 각자
    # 다른 해석(Global max vs 직접 Flatten)으로 재현한 것이므로, 참고용으로만 Variant3의
    # Study2 수치를 그대로 표시 - 엄밀한 비교 기준이 아니라 "두 파일이 공통으로 겨냥하는
    # 목표"라는 의미로만 사용(Run Settings의 note 참조).
    "cnn3d": (93.41, 95.76, 88.98, 91.68),
    # 논문은 ResNet 자체(SoftMax) 분류 정확도를 직접 보고하지 않음 - Table4의
    # "3D-ResNet(FC-4) -> GB 90.0%"는 별도 파이프라인 단계(ResNet 특징을 Gradient
    # Boosting으로 분류)라 이 스크립트가 재현하는 ResNet SoftMax 분류와 직접 비교 불가.
    # 참고용으로만 90.0%를 accuracy 자리에 넣고, precision/recall/f1은 논문 미보고라 None.
    "resnet3d": (90.0, None, None, None),
}

PAPER_LAYER_COUNT = {"base": 8, "variant1": 9, "variant2": 17, "variant3": 24, "cnn3d": 24, "resnet3d": 15}


def load_summary(variant):
    with open(os.path.join(RESULTS_DIR, RUN_FILES[variant]), encoding="utf-8") as f:
        return json.load(f)


def compute_confusion(variant):
    from compute_confusion import analyze
    return analyze(os.path.join(CKPT_DIR, CKPT_FILES[variant]))


def model_structure_rows(variant):
    # [2026-07-26 갱신, 2026-07-27 파일명 갱신] models.py가 cnn3d_figure2.py(1차)/
    # final_cnn.py+final_resnet.py(최종)로 분리됨에 따라 import 경로 갱신.
    from ablation_models import CNN3D_Base, CNN3D_Variant1, CNN3D_Variant2, CNN3D_Variant3
    from final_resnet import ResNet3D
    from cnn3d_figure2 import CNN3D_Figure2
    cls_map = {
        "base": CNN3D_Base, "variant1": CNN3D_Variant1, "variant2": CNN3D_Variant2,
        "variant3": CNN3D_Variant3, "cnn3d": CNN3D_Figure2, "resnet3d": ResNet3D,
    }
    m = cls_map[variant]()
    rows = []
    i = 1
    for lname, mod in m.named_modules():
        if lname == "" or list(mod.children()):
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        rows.append((i, lname, type(mod).__name__, n_params, n_params, repr(mod)))
        i += 1
    return rows


# (order, paper_layer_label, actual_module_classname, detail)
PAPER_ARCH = {
    "base": [
        (1, "Input", "Conv3d", "Conv3d(1,32,kernel=3,stride=1,padding=1) - 논문은 Input을 별도 레이어로 세지만 우리 코드는 conv1이 이를 겸함"),
        (2, "Conv3D", "Conv3d", "conv2: Conv3d(32,64,kernel=3,stride=1,padding=1)"),
        (3, "Activation", "ReLU", "conv 2개 후 1회 적용(논문 원문 단수 표기 'an activation layer')"),
        (4, "Max Pooling", "MaxPool3d", "MaxPool3d(kernel=2,stride=2) - Table3 Study2 육안확인"),
        (5, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(64)"),
        (6, "Flattened Dense", "Flatten + Linear", "Flatten 후 Linear(1404928,3) - Table3 Study2 Flatten=Flatten(그대로) 확인"),
        (7, "SoftMax", "(CrossEntropyLoss 내장)", "Softmax는 학습 loss에 내장, 추론시에만 적용"),
    ],
    "variant1": [
        (1, "Input", "Conv3d", "conv1: Conv3d(1,32,kernel=3,stride=1,padding=1)"),
        (2, "Conv3D", "Conv3d", "conv2: Conv3d(32,64,...)"),
        (3, "Conv3D", "Conv3d", "conv3: Conv3d(64,128,...)"),
        (4, "Activation", "ReLU", "conv 3개 후 1회 적용(논문 원문 단수 표기)"),
        (5, "Average Pooling", "AvgPool3d", "AvgPool3d(kernel=2,stride=2) - Table3 Study2 육안확인 Pooling=Average"),
        (6, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(128)"),
        (7, "Global Max (Flatten 대체)", "AdaptiveMaxPool3d", "Table3 Study2 육안확인 Flatten=Global max - 구조 자체 변경(classifier 입력 2,809,856->128)"),
        (8, "Flattened Dense", "Flatten + Linear", "Flatten 후 Linear(128,3)"),
        (9, "SoftMax", "(CrossEntropyLoss 내장)", "Softmax는 학습 loss에 내장"),
    ],
    "variant2": [
        (1, "Input", "Conv3d", "conv1(블록1): Conv3d(1,32,...)"),
        (2, "Conv3D", "Conv3d", "conv2(블록1): Conv3d(32,64,...)"),
        (3, "Activation", "ReLU", "블록1 conv 2개, ReLU 2회(논문 복수 표기 'activation layers')"),
        (4, "Max Pooling", "MaxPool3d", "블록1 뒤 MaxPool3d"),
        (5, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(64)"),
        (6, "Conv3D", "Conv3d", "conv3(블록2): Conv3d(64,128,...)"),
        (7, "Conv3D", "Conv3d", "conv4(블록2): Conv3d(128,256,...)"),
        (8, "Conv3D", "Conv3d", "conv5(블록2): Conv3d(256,512,...)"),
        (9, "Activation", "ReLU", "블록2 conv 3개, ReLU 3회"),
        (10, "Max Pooling", "MaxPool3d", "블록2 뒤 MaxPool3d"),
        (11, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(512)"),
        (12, "Flattened Dense", "Flatten + Linear", "Flatten 후 Linear(1404928,3) - Study2 Flatten=Flatten 확인"),
        (13, "SoftMax", "(CrossEntropyLoss 내장)", "Softmax는 학습 loss에 내장"),
    ],
    "variant3": [
        (1, "Input", "Conv3d", "conv1(블록1): Conv3d(1,32,...)"),
        (2, "Conv3D", "Conv3d", "conv2(블록1): Conv3d(32,64,...)"),
        (3, "Activation", "ReLU", "블록1 ReLU 2회"),
        (4, "Average Pooling", "AvgPool3d", "블록1 뒤 AvgPool3d - Study2 Pooling=Average"),
        (5, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(64)"),
        (6, "Conv3D", "Conv3d", "conv3~5(블록2): 64->128->256->512"),
        (7, "Activation", "ReLU", "블록2 ReLU 3회"),
        (8, "Average Pooling", "AvgPool3d", "블록2 뒤 AvgPool3d"),
        (9, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(512)"),
        (10, "Conv3D", "Conv3d", "conv6~7(블록3): 512->256->128"),
        (11, "Activation", "ReLU", "블록3 ReLU 2회"),
        (12, "Average Pooling", "AvgPool3d", "블록3 뒤 AvgPool3d"),
        (13, "Batch Normalization", "BatchNorm3d", "BatchNorm3d(128)"),
        (14, "Global Max (Flatten 대체)", "AdaptiveMaxPool3d", "Study2 Flatten=Global max"),
        (15, "FC-1", "Linear", "Linear(128,1000) - 사용자 제공 'Proposed 3D CNN' 다이어그램"),
        (16, "FC-2", "Linear", "Linear(128,1000) - FC-1과 병렬"),
        (17, "SoftMax", "(CrossEntropyLoss 내장)", "Linear(2000,3) 후 Softmax는 학습 loss에 내장"),
    ],
    # [2026-07-24 추가] Figure2(논문 "제안 아키텍처" 최종모델) 직접 재현판. 같은
    # "24-layer 최종모델"을 ablation_models.py CNN3D_Variant3는 Table3 Study2 그리드서치
    # 결과(Global max/batch32/lr0.001)로, 이 CNN3D는 Figure2 다이어그램(채널
    # 32-64-128-256-512-1024-512-256, 직접 Flatten=12544)으로 각각 재현 - 두 파일의
    # 근거가 다름(models.py CNN3D docstring, DEVIATIONS.md 1번/9번 섹션 참조).
    "cnn3d": [
        (1, "Input", "Conv3d", "conv0: Conv3d(1,32,kernel=3,stride=1,padding=1)"),
        (2, "Conv3D", "Conv3d", "conv1: Conv3d(32,64,...), conv2: Conv3d(64,128,...)"),
        (3, "Activation", "ReLU", "conv0,1,2 후 각각 적용"),
        (4, "Max Pooling", "MaxPool3d", "pool1: MaxPool3d(kernel=2,stride=2), 56->28"),
        (5, "Batch Normalization", "BatchNorm3d", "bn1: BatchNorm3d(128)"),
        (6, "Conv3D", "Conv3d", "conv3(128->256), conv4(256->512), conv5(512->1024)"),
        (7, "Activation", "ReLU", "conv3,4,5 후 각각 적용"),
        (8, "Max Pooling", "MaxPool3d", "pool2: MaxPool3d(kernel=2,stride=2), 28->14"),
        (9, "Batch Normalization", "BatchNorm3d", "bn2: BatchNorm3d(1024)"),
        (10, "Conv3D", "Conv3d", "conv6(1024->512), conv7(512->256) - Figure2 마지막 블록 채널 감소 패턴"),
        (11, "Activation", "ReLU", "conv6,7 후 각각 적용"),
        (12, "Max Pooling", "MaxPool3d", "pool3: MaxPool3d(kernel=2,stride=2), 14->7"),
        (13, "Batch Normalization", "BatchNorm3d", "bn3: BatchNorm3d(256)"),
        (14, "Flatten", "AdaptiveAvgPool3d(depth_collapse)+Flatten", "spatial_collapse로 depth축만 1로 축소 후 flatten -> 12544 (Figure2 명시값과 일치, 산술 정확값 87808과는 불일치 - 논문 자체 수치 채택)"),
        (15, "FC-1", "Linear", "Linear(12544,1000) - 본문 p.5 'FC-1: 1000 features'"),
        (16, "FC-2", "Linear", "Linear(12544,1000) - FC-1과 병렬"),
        (17, "SoftMax", "(CrossEntropyLoss 내장)", "Linear(2000,3) 후 Softmax는 학습 loss에 내장. FV-3=2000은 Fig.1 라벨 기준(본문 Eq.2는 elementwise max=1000과 불일치 - 논문 자체 모순, DEVIATIONS.md 참조)"),
    ],
    # [2026-07-23 추가] Figure3 원문: "composed of 15 layers, including an input layer,
    # two conv3D blocks, five residual blocks with max-pooling layers, a fully connected
    # layer, and a SoftMax layer... residual block contains three residual units" ->
    # 15 = 5 Residual Block x 3 unit (models.py 상단 docstring과 동일 해석).
    "resnet3d": [
        (1, "Input + Conv3D Block 1", "Conv3d+BatchNorm3d+ReLU", "stem[0:3]: Conv3d(1,32)+BN+ReLU - 논문은 Input을 별도 레이어로 세지만 stem 첫 conv가 이를 겸함"),
        (2, "Conv3D Block 2", "Conv3d+BatchNorm3d+ReLU", "stem[3:6]: Conv3d(32,64)+BN+ReLU"),
        (3, "Residual Block 1 (3 unit)", "ResidualBlock3D(64->64)", "block0: unit0~2, 각 unit=Conv3d+BN+ReLU+Conv3d+BN+skip+ReLU, 뒤에 MaxPool3d(pool0)"),
        (4, "Residual Block 2 (3 unit)", "ResidualBlock3D(64->128)", "block1: 64->128 채널 확장(unit0 skip에 1x1 conv), 뒤에 MaxPool3d(pool1)"),
        (5, "Residual Block 3 (3 unit)", "ResidualBlock3D(128->256)", "block2: 128->256, 뒤에 MaxPool3d(pool2)"),
        (6, "Residual Block 4 (3 unit)", "ResidualBlock3D(256->512)", "block3: 256->512, 뒤에 MaxPool3d(pool3)"),
        (7, "Residual Block 5 (3 unit)", "ResidualBlock3D(512->512)", "block4: 512->512(채널 유지) - 논문 Figure3에 채널 수 전혀 미기재, 표준 ResNet 더블링 관행을 4단계까지만 적용하고 마지막은 유지(자체결정), 뒤에 MaxPool3d(pool4, 2026-07-22 AdaptiveAvgPool3d에서 MaxPool3d로 정정)"),
        (8, "Fully Connected", "Linear", "fc4: Linear(512,1000) - 본문 '1000 features are extracted from the 14th layer'"),
        (9, "SoftMax", "(CrossEntropyLoss 내장)", "classifier: Linear(1000,3) 후 Softmax는 학습 loss에 내장"),
    ],
}

ASSUMPTIONS_COMMON = [
    ("paper_disclosed", "dataset_size", 303),
    ("paper_disclosed", "class_counts", '{"Control": 110, "PD": 135, "Prodromal": 58}'),
    ("paper_disclosed", "input_shape", "[1, 56, 56, 56]"),
    ("paper_disclosed", "split", "70% train, 15% validation, 15% test"),
    ("paper_disclosed", "optimizer", "Adam (Table5 'Optimization: Grid search' 결과 최종 채택값)"),
    ("paper_disclosed", "loss", "categorical cross-entropy"),
    ("paper_disclosed", "epochs", 30),
    # [2026-07-23 갱신] Table5를 다시 확인한 결과 weight_decay/k-fold 둘 다 논문에 명시돼
    # 있었음(이전 기록이 "자체결정"/"미적용"으로 잘못 남아있었음) - 정정.
    ("paper_disclosed", "weight_decay", "있음 (Table5 'L2 Regularization'=0.0001, CNN/ResNet 공통) - 자체결정 아님, 논문 명시값 그대로 사용"),
    ("paper_disclosed", "fivefold_cross_validation", "있음 (Table5 'k-fold CV'=5, CNN/ResNet 공통) - train_ablation.py/train_resnet.py --kfold 5 옵션으로 지원(2026-07-23 추가). 기본 실행은 여전히 single holdout 병행(비교 기준선 유지 목적)"),
    ("paper_disclosed", "data_augmentation", "없음 (Methods에 언급만 있고 구체 기법 미기재 - 우리는 미구현)"),
    ("project_decisions", "random_seed", 42),
    ("project_decisions", "split_method", "subject-level 70/15/15 holdout (dataset.py get_holdout_split, 기본 실행) / k-fold 실행시 get_kfold_splits_with_val (Table5 k=5 재현, 2026-07-23 추가 - fold 내부 val 분리 방식은 자체결정, dataset.py docstring 참조)"),
    ("project_decisions", "checkpoint_metric", "기본값: validation loss 최저 시점(2026-07-21 변경) - val_acc는 45개 검증샘플 기준 양자화가 심해 오판 유발 확인됨. 단, variant3 최종 체크포인트는 팀원 J와 동일 기준(val_acc 최고)으로 재학습해 클래스 붕괴 해결 - 모델별 실제 값은 Run Settings 시트 checkpoint_selection 참조"),
    ("project_decisions", "gradient_clip_norm", "1.0 - 논문 미기재, classifier 로짓 폭주 방지용 자체결정"),
    ("project_decisions", "preprocessing", "01_Preprocessing/전처리_ref21order_v1 (BET->ANTsPy 정합->min-max 정규화->56^3 리사이즈, N4 비활성)"),
]

# ResNet 전용 assumptions(4개 CNN variant와 다른 항목만)
ASSUMPTIONS_RESNET_ONLY = [
    ("paper_disclosed", "resnet_channel_width", "전혀 미기재(Figure3에 수치 없음) - 64->128->256->512->512(표준 더블링, 마지막 유지) 자체 적용"),
    ("paper_disclosed", "resnet_learning_rate", "있음 (Table5 3D ResNet Learning rate=0.0001, CNN의 0.001과 별도 명시) - 2026-07-22 정정 완료(과거 lr=0.001로 5.5시간 낭비한 이력 있음)"),
    ("paper_disclosed", "resnet_own_accuracy_not_reported", "논문은 ResNet 단독 SoftMax 분류 정확도를 직접 보고하지 않음 - Table4의 'ResNet FC-4->GB 90.0%'는 별도 파이프라인(GB 분류기) 결과라 이 리포트의 test_accuracy와 직접 비교 불가(참고용)"),
]

# [2026-07-24 추가] CNN3D(models.py, Model-1 최종모델) 전용 assumptions
ASSUMPTIONS_CNN3D_ONLY = [
    ("paper_disclosed", "cnn3d_batch_size", "미기재(Table3 Study2 그리드서치 대상 아님) - batch=32 자체 선택(Variant3와 동일값으로 통일)"),
    ("paper_disclosed", "cnn3d_flatten_depth_collapse", "Figure2 명시(12544) vs 산술 정확값(87808) 불일치 - AdaptiveAvgPool3d로 depth축만 축소해 12544 재현(자체 해석)"),
    ("paper_disclosed", "cnn3d_fv3_dim", "본문 Eq.2(elementwise max=1000) vs Fig.1 라벨(x2000) 불일치 - concatenation(2000차원) 채택, 논문 자체 모순(DEVIATIONS.md 1번 섹션)"),
    ("paper_disclosed", "cnn3d_vs_variant3_not_target_accuracy", "이 표의 Paper Accuracy(93.41% 등)는 CNN3D_Variant3의 Study2 보고값을 참고용으로 그대로 표시한 것 - CNN3D는 별도 그리드서치 대상이 아니라 엄밀한 비교 기준 아님"),
    ("project_decisions", "cnn3d_class_collapse_2026-07-24", "batch=32, 30epoch(24epoch 조기종료) 학습 결과 전 샘플을 PD로만 예측하는 완전한 클래스 붕괴 발생(Test Accuracy 45.65%=PD 비율과 동일, Precision macro 15.56%) - class-weighted loss 등 추가 조치 필요, 원인 미해결"),
]


def main():
    shutil.copyfile(TEMPLATE_FILE, OUT_FILE)
    wb = openpyxl.load_workbook(OUT_FILE)

    summaries = {v: load_summary(v) for v in ALL_VARIANTS}
    confusions = {v: compute_confusion(v) for v in ALL_VARIANTS}
    structures = {v: model_structure_rows(v) for v in ALL_VARIANTS}

    def clear_data_rows(ws):
        orig_max_row = ws.max_row
        orig_max_col = ws.max_column
        ref_formats = [ws.cell(row=2, column=c).number_format for c in range(1, orig_max_col + 1)]
        for r in range(2, orig_max_row + 1):
            for c in range(1, orig_max_col + 1):
                ws.cell(row=r, column=c).value = None
        return orig_max_row, ref_formats

    def apply_ref_format(ws, row, ref_formats):
        for c, fmt in enumerate(ref_formats, start=1):
            ws.cell(row=row, column=c).number_format = fmt

    # ================= Model Summary =================
    ws = wb["Model Summary"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        s = summaries[v]["summary"]
        hp = s["hyperparams"]
        cm = confusions[v]
        n_params = sum(x[3] for x in structures[v])
        paper = PAPER_ACC[v]
        n_layers = PAPER_LAYER_COUNT[v]
        model_type = MODEL_TYPE[v]
        row = [
            DISPLAY_NAME[v], model_type, v, n_params,
            n_layers, n_layers, True,
            hp["batch_size"], hp["batch_size"], hp["batch_size"], hp["lr"],
            s["best_epoch"], s["test_accuracy"], s["test_precision_macro"],
            s["test_recall_macro"], s["test_f1_macro"],
            cm["precision_weighted"], cm["recall_weighted"], cm["f1_weighted"],
            (paper[0] / 100 if paper[0] is not None else None),
            (paper[1] / 100 if paper[1] is not None else None),
            (paper[2] / 100 if paper[2] is not None else None),
            (paper[3] / 100 if paper[3] is not None else None),
            os.path.join(CKPT_DIR, CKPT_FILES[v]),
        ]
        if r > orig_max_row:
            apply_ref_format(ws, r, ref_formats)
        for ci, val in enumerate(row, start=1):
            ws.cell(row=r, column=ci, value=val)
        r += 1
    print("Model Summary: done")

    # ================= Split Counts =================
    from dataset import get_holdout_split
    import collections
    ws = wb["Split Counts"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    csv_path = os.path.join(_ROOT, "01_Preprocessing", "data_0713_wsl_v2.csv")
    train, val, test = get_holdout_split(csv_path, seed=42)
    for v in ALL_VARIANTS:
        for split_name, split in [("train", train), ("validation", val), ("test", test)]:
            c = collections.Counter(s["Group"] for s in split)
            row = [DISPLAY_NAME[v], split_name, c.get("Control", 0), c.get("Prodromal", 0), c.get("PD", 0), len(split)]
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            for ci, val_ in enumerate(row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    print("Split Counts: done")

    # ================= Model Structure =================
    ws = wb["Model Structure"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        for (idx, lname, ltype, params, trainable, defn) in structures[v]:
            row = [DISPLAY_NAME[v], idx, lname, ltype, params, trainable, defn]
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            for ci, val_ in enumerate(row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    print("Model Structure: done")

    # ================= Paper Architecture =================
    ws = wb["Paper Architecture"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        for (order, layer, actual_module, detail) in PAPER_ARCH[v]:
            row = [DISPLAY_NAME[v], order, layer, actual_module, detail]
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            for ci, val_ in enumerate(row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    print("Paper Architecture: done")

    # ================= Confusion Matrix =================
    ws = wb["Confusion Matrix"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        cm = confusions[v]
        names = cm["class_names"]
        mat = cm["confusion_matrix"]
        for i, true_cls in enumerate(names):
            row = [DISPLAY_NAME[v], true_cls] + [mat[i][j] for j in range(len(names))]
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            for ci, val_ in enumerate(row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    print("Confusion Matrix: done")

    # ================= Run Settings =================
    ws = wb["Run Settings"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        s = summaries[v]["summary"]
        hp = s["hyperparams"]
        paper = PAPER_ACC[v]
        # [2026-07-26 갱신] class_weighted/sampler_weighted/checkpoint_metric은
        # 2026-07-25에 train_ablation.py에 추가된 옵션이라 그 이전 결과 JSON에는
        # 필드 자체가 없음 - hp.get()으로 없으면 그 시절 실제 동작(비가중/val_loss
        # 기준)을 그대로 반영, 있으면(신규 Variant3 등) 실제 값을 그대로 표시.
        class_weighted = hp.get("class_weighted", False)
        sampler_weighted = hp.get("sampler_weighted", False)
        checkpoint_metric = hp.get("checkpoint_metric", "val_loss")
        loss_desc = (
            f"CrossEntropyLoss(class_weighted={class_weighted}"
            + (f", weights={hp.get('class_weights')}" if class_weighted else "")
            + ")"
        )
        checkpoint_desc = (
            "highest validation accuracy (tie-break: lowest validation loss)"
            if checkpoint_metric == "val_acc" else "lowest validation loss"
        )
        settings = [
            ("paper_batch_size", hp["batch_size"]), ("learning_rate", hp["lr"]),
            ("epochs", hp["epochs"]), ("weight_decay", hp["weight_decay"]),
            ("paper_accuracy", paper[0]), ("paper_precision", paper[1]),
            ("paper_recall", paper[2]), ("paper_f1", paper[3]),
            ("physical_batch_size", hp["batch_size"]), ("accumulation_steps", 1),
            ("effective_batch_size", hp["batch_size"]),
            ("batch_mode", "direct (gradient accumulation 미사용)"),
            ("optimizer", "Adam"),
            ("loss", loss_desc),
            ("amp", False), ("scheduler", "none"),
            ("gradient_clipping", hp["grad_clip_norm"]),
            ("class_weighting", "inverse frequency (train 분포 기준)" if class_weighted else "none"),
            ("sampler", "WeightedRandomSampler (역빈도, 복원추출)" if sampler_weighted else "none (shuffle=True)"),
            ("label_smoothing", 0), ("dropout", 0),
            ("checkpoint_selection", checkpoint_desc),
            ("patience", hp["patience"]), ("min_epochs", hp["min_epochs"]),
            ("random_seed", hp["seed"]), ("calibrate_init", hp["calibrate_init"]),
        ]
        if v == "resnet3d":
            settings.append(("note", "paper_accuracy는 Table4 GB(FC-4) 결과(90.0%) 참고용 - 이 표의 test_accuracy(ResNet SoftMax 직접분류)와 파이프라인 단계가 달라 직접비교 불가"))
        if v == "cnn3d":
            settings.append(("note", "1차 모델(Figure2 직접 재현, cnn3d_figure2.py) - 2026-07-24 학습 결과 전 샘플 PD 예측(완전 클래스 붕괴), 이후 재학습 안 함. 최종 모델은 variant3(final_cnn.py의 CNN3D) 참조"))
        if v == "variant3":
            settings.append(("note", "2026-07-26: class_weighted+sampler+val_acc 체크포인트 기준 조합으로 재학습해 클래스 붕괴 해결(이전 시도는 Prodromal recall 항상 0% - 세션_기록_클래스붕괴_대응_ClassWeight_Sampler.md 참조). final_cnn.py의 CNN3D가 이 체크포인트의 별칭"))
        for key, val_ in settings:
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            ws.cell(row=r, column=1, value=DISPLAY_NAME[v])
            ws.cell(row=r, column=2, value=key)
            ws.cell(row=r, column=3, value=val_)
            r += 1
    print("Run Settings: done")

    # ================= Assumptions =================
    ws = wb["Assumptions"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        rows_for_v = list(ASSUMPTIONS_COMMON)
        if v == "resnet3d":
            rows_for_v = rows_for_v + ASSUMPTIONS_RESNET_ONLY
        if v == "cnn3d":
            rows_for_v = rows_for_v + ASSUMPTIONS_CNN3D_ONLY
        for (cat, key, val_) in rows_for_v:
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            ws.cell(row=r, column=1, value=DISPLAY_NAME[v])
            ws.cell(row=r, column=2, value=cat)
            ws.cell(row=r, column=3, value=key)
            ws.cell(row=r, column=4, value=val_)
            r += 1
    print("Assumptions: done")

    # ================= Epoch Metrics =================
    # [2026-07-23 변경] "없음" 하드코딩 대신 h.get(key, "없음") - 새 형식(precision/recall/f1
    # 포함) JSON이면 자동으로 값이 채워지고, 구형식 JSON이면 그대로 "없음" 유지.
    # [2026-07-24 추가] weighted 지표(train/val 각각 precision/recall/f1)도 동일한 방식
    # 으로 h.get() - train_loop()이 2026-07-24부터 이 필드들을 기록하므로, 이 스크립트를
    # 다시 돌린 "이후" 재학습 결과부터는 자동으로 채워짐(과거 JSON은 "없음" 유지 -
    # epoch별 중간 가중치가 저장돼 있지 않아 소급 계산이 불가능함).
    ws = wb["Epoch Metrics"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        hist = summaries[v]["history"]
        lr = summaries[v]["summary"]["hyperparams"]["lr"]
        for h in hist:
            row = [
                DISPLAY_NAME[v], h["epoch"], h["epoch_seconds"], lr,
                h["train_loss"], h["train_acc"],
                h.get("train_precision_macro", "없음"), h.get("train_recall_macro", "없음"),
                h.get("train_f1_macro", "없음"),
                h.get("train_precision_weighted", "없음"), h.get("train_recall_weighted", "없음"),
                h.get("train_f1_weighted", "없음"),
                h["val_loss"], h["val_acc"],
                h.get("val_precision_macro", "없음"), h.get("val_recall_macro", "없음"),
                h.get("val_f1_macro", "없음"),
                h.get("val_precision_weighted", "없음"), h.get("val_recall_weighted", "없음"),
                h.get("val_f1_weighted", "없음"),
            ]
            if r > orig_max_row:
                apply_ref_format(ws, r, ref_formats)
            for ci, val_ in enumerate(row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    print("Epoch Metrics: done")

    # ================= Best Validation =================
    # [2026-07-24 재작성] 기존 코드는 history의 best_row에서 값을 끌어왔는데, (1) 과거
    # JSON엔 weighted 지표 자체가 없어서 전부 "없음"이었고 (2) 컬럼 매핑에 버그가 있었음
    # (Train F1 Macro 자리에 train_precision_macro가, Train F1 Weighted 자리에
    # train_f1_macro가 잘못 들어가고 있었음). 재학습 없이도 채울 수 있도록, 저장된
    # 체크포인트(=best epoch 가중치) 그대로를 train/val 세트에 다시 추론시켜서
    # macro+weighted 지표와 클래스별 예측개수를 직접 계산 - 6개 모델 전부 즉시 채워짐.
    from compute_confusion import analyze_split
    ws = wb["Best Validation"]
    orig_max_row, ref_formats = clear_data_rows(ws)
    r = 2
    for v in ALL_VARIANTS:
        s = summaries[v]["summary"]
        hist = summaries[v]["history"]
        best_ep = s["best_epoch"]
        best_row = next((h for h in hist if h["epoch"] == best_ep), None)
        ckpt_path = os.path.join(CKPT_DIR, CKPT_FILES[v])
        train_eval = analyze_split(ckpt_path, split="train")
        val_eval = analyze_split(ckpt_path, split="val")

        train_acc = train_eval["accuracy"]
        train_f1_macro = train_eval["f1_macro"]
        train_f1_weighted = train_eval["f1_weighted"]
        val_loss = best_row["val_loss"] if best_row else "없음"
        val_acc = val_eval["accuracy"]
        acc_gap = train_acc - val_acc
        macro_f1_gap = train_f1_macro - val_eval["f1_macro"]
        weighted_f1_gap = train_f1_weighted - val_eval["f1_weighted"]
        pred_counts_str = json.dumps(val_eval["pred_counts"], ensure_ascii=False)

        row = [
            DISPLAY_NAME[v], best_ep, train_acc, train_f1_macro, train_f1_weighted,
            val_loss, val_acc,
            val_eval["precision_macro"], val_eval["recall_macro"], val_eval["f1_macro"],
            val_eval["precision_weighted"], val_eval["recall_weighted"], val_eval["f1_weighted"],
            acc_gap, macro_f1_gap, weighted_f1_gap, pred_counts_str,
        ]
        if r > orig_max_row:
            apply_ref_format(ws, r, ref_formats)
        for ci, val_ in enumerate(row, start=1):
            ws.cell(row=r, column=ci, value=val_)
        r += 1
    print("Best Validation: done (체크포인트 재추론으로 train/val 지표 직접 계산)")

    # ================= K-Fold Results (신규 시트) =================
    # [2026-07-23 추가] results/kfold_*.json(train_ablation.py --kfold / train_resnet.py
    # --kfold 실행 결과)을 자동으로 스캔해서 채움. 아직 하나도 안 돌렸으면 헤더만 있는
    # 빈 시트가 생성됨 - k-fold를 실행한 뒤 이 스크립트를 다시 돌리기만 하면 자동 반영.
    if "K-Fold Results" in wb.sheetnames:
        del wb["K-Fold Results"]
    ws = wb.create_sheet("K-Fold Results")
    headers = [
        "Model", "K", "Fold", "N Params", "Actual Epochs", "Best Epoch",
        "Best Val Acc", "Test Accuracy", "Test Precision(macro)", "Test Recall(macro)",
        "Test F1(macro)", "Elapsed(sec)", "Row Type",
    ]
    for ci, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")

    kfold_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "kfold_*.json")))
    r = 2
    if not kfold_files:
        ws.cell(row=r, column=1, value="(아직 k-fold 실행 결과 없음 - train_ablation.py/train_resnet.py에 --kfold 5(또는 10/15) 옵션으로 실행 후 이 스크립트를 다시 실행하면 자동으로 채워짐)")
        r += 1
    else:
        for fpath in kfold_files:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            s = d["summary"]
            model_name = DISPLAY_NAME.get(s["variant"], s["variant"])
            k = s["kfold"]
            for fs in d["fold_summaries"]:
                row = [
                    model_name, k, fs["fold"], fs["n_params"], fs["actual_epochs"], fs["best_epoch"],
                    fs["best_val_acc"], fs["test_accuracy"], fs["test_precision_macro"],
                    fs["test_recall_macro"], fs["test_f1_macro"], fs["elapsed_sec"], "fold",
                ]
                for ci, val_ in enumerate(row, start=1):
                    ws.cell(row=r, column=ci, value=val_)
                r += 1
            agg_row = [
                model_name, k, "MEAN", "-", "-", "-", "-",
                s["test_accuracy_mean"], s["test_precision_macro_mean"],
                s["test_recall_macro_mean"], s["test_f1_macro_mean"], s["elapsed_sec"], "aggregate_mean",
            ]
            for ci, val_ in enumerate(agg_row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
            std_row = [
                model_name, k, "STD", "-", "-", "-", "-",
                s["test_accuracy_std"], s["test_precision_macro_std"],
                s["test_recall_macro_std"], s["test_f1_macro_std"], "-", "aggregate_std",
            ]
            for ci, val_ in enumerate(std_row, start=1):
                ws.cell(row=r, column=ci, value=val_)
            r += 1
    for ci in range(1, len(headers) + 1):
        ws.column_dimensions[chr(64 + ci) if ci <= 26 else "A"].width = 16
    print(f"K-Fold Results: done ({len(kfold_files)}개 집계 파일 반영)")

    wb.save(OUT_FILE)
    print("SAVED:", OUT_FILE)

    # [2026-07-24 추가] Downloads는 언제든 정리될 수 있는 임시 폴더라, 다른 원본 결과
    # (results/*.json, checkpoints/*.pt)와 달리 이 요약 Excel만 프로젝트 밖에만
    # 있었음 - reports/ 폴더에도 동일 파일을 사본으로 남겨서 버전 관리되게 함.
    reports_dir = os.path.join(_ROOT, "03_Model_Training", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    project_copy_path = os.path.join(reports_dir, os.path.basename(OUT_FILE))
    shutil.copyfile(OUT_FILE, project_copy_path)
    print("프로젝트 폴더에도 사본 저장:", project_copy_path)


if __name__ == "__main__":
    main()
