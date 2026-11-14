# ---------------------------------------------------------------
# 재활용품 이미지 분류용 PyTorch 커스텀 Dataset 정의 파일
# (폴더 구조: 데이터셋 폴더/클래스명/이미지.jpg)
# ---------------------------------------------------------------
import os  # 폴더/파일 경로 다루기, 시스템 명령 실행용
from torch.utils.data import Dataset  # 커스텀 데이터셋이 상속받는 PyTorch 기본 클래스
import torchvision.transforms as transforms  # 이미지 전처리(리사이즈, 텐서 변환 등) 도구 모음
from PIL import Image  # 이미지 파일을 열고 모드를 변환하기 위한 라이브러리

class Pytorch_Custom_Dataset_class(Dataset):
    def __init__(self):
        super().__init__()  # 부모(Dataset) 클래스 초기화
        pass

    def __getitem__(self):  # 데이터 1개를 꺼내는 함수
        pass
    def __len__(self):  # 전체 데이터 개수를 반환하는 함수
        pass


def __init__(
        self, dataset_dir="/content/Recycle_Classification_Dataset", transform=None

):
    # 데이터셋 폴더가 없으면 깃허브에서 내려받음
    if not os.path.isdir(dataset_dir):
        os.system(
            # 올바른 주소로 다시 clone
            "git clone https://github.com/JinFree/Recycle_Classification_Dataset.git"
        )
        # 불필요한 .git 폴더 삭제 (경로 대소문자가 clone된 폴더명과 달라 삭제 안 될 수 있음)
        os.system("rm -rf ./recycle_Classification_Dataset/.git")
    self.image_abs_path = dataset_dir  # 이미지가 들어있는 최상위 폴더 경로 저장

    self.transform = transform  # 외부에서 전달한 전처리 방식 저장
    if self.transform is None:  # 전달된 게 없으면 기본 전처리 사용
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),  # 짧은 변 길이를 256px로 맞춤
                transforms.RandomCrop(224),  # 224x224 크기로 무작위 자르기 (데이터 증강 효과)
                transforms.ToTensor(),  # PIL 이미지 -> 텐서 변환 (픽셀값 0~255 -> 0~1)
                transforms.Normalize(
                    # ImageNet 학습 데이터의 채널별(R,G,B) 평균/표준편차로 정규화
                    mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

        # [주의] 아래 코드가 if 블록 안에 들어가 있어서 transform을 직접 넘기면 실행되지 않음
        self.label_list = os.listdir(self.image_abs_path)  # 하위 폴더명 = 클래스(라벨) 이름 목록

        self.label_list.sort()  # 항상 같은 순서로 라벨 번호가 매겨지도록 정렬
        self.x_list = []  # 이미지 파일 경로를 담을 리스트
        self.y_list = []  # 각 이미지의 정답 라벨(정수)을 담을 리스트

        # 클래스 폴더를 하나씩 돌면서 (번호, 이름) 획득
        for label_index, label_str in enumerate(self.label_list):
            img_path = os.path.join(self.image_abs_path, label_str)  # 클래스 폴더의 전체 경로
            img_list = os.listdir(img_path)  # 해당 폴더 안의 이미지 파일명 목록

            for img in img_list:
                self.x_list.append(os.path.join(img_path, img))  # 이미지 전체 경로 저장
                self.y_list.append(label_index)  # 그 이미지의 라벨 번호 저장

def __len__(self):
    return len(self.x_list)  # 전체 이미지 개수

def __getitem__(self, idx):
    image = Image.open(self.x_list[idx])  # idx번째 이미지 파일 열기

    # 흑백(L), 투명(RGBA) 등 다른 모드면 3채널 RGB로 통일 (모델 입력 채널 수를 맞추기 위함)
    if image.mode != "RGB":
        image = image.convert('RGB')

    # 전처리가 지정되어 있으면 적용
    if self.transform is not None:
        image = self.transform(image)

    return image, self.y_list[idx]  # (이미지 텐서, 정답 라벨) 쌍 반환

# ---------------------------------------------------------------
# [완성본] 실제로 사용할 재활용품 분류 데이터셋 클래스
# DataLoader에 넣어 배치 단위로 학습/평가할 수 있음
# ---------------------------------------------------------------
class PyTorch_Classification_Dataset_Class(Dataset):
    def __init__(
            self, dataset_dir="/content/Recycle_Classification_Dataset", transform =None
    ):
        # 데이터셋 폴더가 없으면 깃허브에서 clone
        if not os.path.isdir(dataset_dir):
            os.system(
                "git clone https://github.com/JinFree/Recycle_Classification_dataset.git"
            )

            # 용량/불필요한 깃 기록 제거
            os.system("rm -rf ./Recycle_Classification_Dataset/.git")
            self.image_abs_path = dataset_dir  # 이미지 최상위 폴더 경로
            self.transform = transform  # 사용자 지정 전처리

            # 전처리가 없으면 기본 전처리(리사이즈 -> 랜덤크롭 -> 텐서 -> 정규화) 사용
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(256),  # 짧은 변을 256px로
                    transforms.RandomCrop(224),  # 224x224 무작위 크롭
                    transforms.ToTensor(),  # 텐서로 변환
                    transforms.Normalize(
                        # ImageNet 기준 채널별 평균/표준편차
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                ])

            # [주의] 여기부터 전부 if not isdir 블록 안이라, 폴더가 이미 있으면 실행되지 않아
            # x_list 등이 만들어지지 않음 (len() 호출 시 에러 발생)
            self.label_list = os.listdir(self.image_abs_path)  # 클래스 이름 = 하위 폴더명
            self.label_list.sort()  # 라벨 번호가 항상 동일하도록 정렬
            self.x_list = []  # 이미지 경로 리스트
            self.y_list = []  # 라벨 번호 리스트
            for label_index, label_str in enumerate(self.label_list):
                img_path = os.path.join(self.image_abs_path, label_str)  # 클래스 폴더 경로
                img_list = os.listdir(img_path)  # 클래스 폴더 내 이미지 목록
                for img in img_list:
                    self.x_list.append(os.path.join(img_path, img))  # 이미지 경로 저장
                    self.y_list.append(label_index)  # 해당 이미지의 라벨 저장

    def __len__(self):
        # DataLoader가 전체 데이터 개수를 알기 위해 호출
        return len(self.x_list)

    def __getitem__(self,idx):
        # DataLoader가 idx번째 샘플을 요청할 때 호출
        image = Image.open(self.x_list[idx])  # 이미지 열기
        if image.mode != "RGB":
            image = image.convert('RGB')  # 3채널 RGB로 통일
        if self.transform is not None:
            image = self.transform(image)  # 전처리 적용
        return image, self.y_list[idx]  # (이미지 텐서, 라벨 번호)

    def __save_label_map__(self, dst_text_path="label_map.txt"):
        # 라벨 번호 순서대로 클래스 이름을 텍스트 파일에 한 줄씩 저장
        # (추론 시 예측 번호 -> 클래스 이름 변환에 사용)
        label_list = self.label_list
        f = open(dst_text_path, 'w')

        for i in range(len(label_list)):
            f.write(label_list[i] + '\n')

        f.close()  # 파일 닫기

    def __num_classes__(self):
        # 전체 클래스(분류 종류) 개수 반환 -> 모델 출력 노드 수 설정에 사용
        return len(self.label_list)
