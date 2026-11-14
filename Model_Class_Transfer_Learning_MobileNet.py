# ---------------------------------------------------------------
# 전이학습(Transfer Learning) 기반 재활용품 분류 모델
# ImageNet으로 미리 학습된 MobileNetV2를 가져와서
# 마지막 분류층만 우리 클래스 개수에 맞게 교체해 사용함
# ---------------------------------------------------------------
import torch  # PyTorch 기본 라이브러리
from torchvision import models  # 사전학습된 유명 모델(MobileNet, ResNet 등) 제공
import torch.nn as nn  # 레이어와 nn.Module 제공
import torch.nn.functional as F  # 함수형 연산 모음 (현재 코드에서는 사용 안 함)


class MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        # num_classes: 분류할 종류 수 / pretrained: ImageNet 사전학습 가중치 사용 여부
        super().__init__()  # 부모(nn.Module) 초기화

        # MobileNetV2 불러오기 (가볍고 빨라서 모바일/소규모 환경에 적합)
        # pretrained=True면 이미 수백만 장으로 학습된 가중치를 그대로 가져옴
        # (최신 torchvision에서는 pretrained 대신 weights= 사용을 권장하며 경고가 뜰 수 있음)
        self.network = models.mobilenet_v2(pretrained=pretrained)

        # 기존 분류층의 입력 특징 개수를 가져옴 (MobileNetV2는 1280)
        # classifier[0]은 Dropout, classifier[1]이 최종 Linear 층
        num_ftrs = self.network.classifier[1].in_features

        # 원래는 ImageNet 1000개 클래스를 출력하던 층을
        # 우리 데이터의 클래스 개수(num_classes)를 출력하는 새 층으로 교체
        self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # 출력 점수를 0~1 확률(합계 1)로 바꿔주는 Softmax를 별도 블록으로 둠
        self.classifier = nn.Sequential(nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.network(x)  # 이미지 -> 특징 추출 -> 클래스별 점수(로짓)
        x = self.classifier(x)  # 점수 -> 확률로 변환
        return x  # 클래스별 확률 반환
