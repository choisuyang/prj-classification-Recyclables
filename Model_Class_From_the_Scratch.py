# ---------------------------------------------------------------
# 재활용품 분류용 CNN 모델을 처음부터(From Scratch) 직접 만든 파일
# 입력: 3채널(RGB) 이미지 텐서 (예: 3x224x224)
# 출력: 클래스별 확률 (합이 1)
# ---------------------------------------------------------------
import torch  # PyTorch 기본 라이브러리 (텐서 연산)
import torch.nn as nn  # 레이어(Conv, Linear 등)와 nn.Module 제공
import torch.nn.functional as F  # 함수형 연산 모음 (현재 코드에서는 사용 안 함)


# [초안/뼈대] 모델 클래스의 기본 구조만 잡아둔 상태
class PyTorch_Custom_Model_Class(nn.Module):
    def __init__(self):
        super().__init__()  # 부모(nn.Module) 초기화 - 모델 클래스에서 반드시 필요
        pass

    def forward(self, x):
        return x  # 입력을 그대로 반환 (아직 아무 연산도 안 함)


# [완성본] 합성곱(Conv) 3단 + 완전연결(Linear) 3단으로 이루어진 분류 모델
class MODEL_From_Scratch(nn.Module):
    def __init__(self, num_classes):
        # num_classes: 분류할 종류 수 (예: 재활용품 종류가 6개면 6)
        super().__init__()
        self.classifier = nn.Sequential(  # 레이어를 순서대로 이어 붙여 하나의 묶음으로 만듦

            # ===== 1단계: 특징 추출 (Conv 블록 1) =====
            # 입력 3채널(RGB) -> 32채널, 3x3 필터, stride=2로 가로세로 크기를 절반으로 줄임
            # padding=1은 가장자리를 0으로 채워 크기 계산을 맞추기 위함
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(32)  # 채널별 값 분포를 정규화 -> 학습 안정화/속도 향상
            , nn.ReLU()  # 활성화 함수: 음수는 0으로 (비선형성 부여)

            # ===== 2단계: Conv 블록 2 =====
            # 32 -> 64채널, 다시 크기 절반 (224 -> 112 -> 56 정도로 작아짐)
            , nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()

            # ===== 3단계: Conv 블록 3 =====
            # 64 -> 128채널, 크기 또 절반. 채널이 늘수록 더 복잡한 특징을 잡음
            , nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(128)
            , nn.ReLU()

            # ===== 특징맵 -> 벡터 변환 =====
            , nn.AdaptiveAvgPool2d(1)  # 각 채널을 평균내어 1x1로 압축 (입력 이미지 크기와 무관하게 128개 값만 남음)
            , nn.Flatten()  # (배치, 128, 1, 1) -> (배치, 128) 형태로 펼침

            # ===== 분류기 (완전연결층) =====
            , nn.Linear(128, 512)  # 128개 특징 -> 512개로 확장
            , nn.ReLU()
            , nn.Dropout()  # 학습 중 일부 뉴런을 무작위로 꺼서 과적합 방지 (기본 확률 0.5)
            , nn.Linear(512, 64)  # 512 -> 64로 축소
            , nn.ReLU()
            , nn.Dropout()
            , nn.Linear(64, num_classes)  # 최종: 클래스 개수만큼의 점수(로짓) 출력
            , nn.Softmax(dim=-1)  # 점수를 0~1 확률로 변환 (합계 1)
        )

    def forward(self, x):
        # 모델에 입력을 넣었을 때 실행되는 순전파: 위에서 만든 레이어를 순서대로 통과
        return self.classifier(x)
