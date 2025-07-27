import os
import random
import time
import torch
import torch.nn as nn
import mujoco
from mujoco import viewer
import msvcrt  # Windows 키 입력 라이브러리
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, VGG16_Weights, VGG19_Weights
import glob
from pathlib import Path

DATA_ROOT = "./cat_gest" 
V10_MODEL_PATH = "./best_hand_gesture_mobilenet_model.pth"
VERA_MODEL_PATH = "./best_hand_gesture_mobilenet_model2.pth"
HAND_MODEL_PATH = "./mujoco_menagerie-main/shadow_hand/scene_right_no_obj.xml"

CLASSES = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HandGestureModel(nn.Module):
    def __init__(self, model_type='mobilenet', num_classes=NUM_CLASSES):
        super(HandGestureModel, self).__init__()
        
        if model_type == 'resnet18':
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'mobilenet':
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
            # print(self.model)
        elif model_type == 'vgg16':
            self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_type == 'vgg19':
            self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_model(model_path, model_type='mobilenet'):
    try:
        model = HandGestureModel(model_type=model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

def load_mujoco_model(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"MuJoCo 모델 파일을 찾을 수 없습니다: {model_path}")
            return None, None
        
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("MuJoCo 모델 로드 성공!")
        return model, data
    except Exception as e:
        print(f"MuJoCo 모델 로드 중 오류 발생: {e}")
        return None, None

def get_random_image():
    all_images = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                all_images.append((img_path, class_name))
    
    if not all_images:
        print("이미지를 찾을 수 없습니다.")
        return None, None
    
    img_path, true_class = random.choice(all_images)
    return img_path, true_class

def predict_gesture(model, image_path):
    try:
        image = Image.open(image_path).convert('L')
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            
        predicted_class = CLASSES[predicted.item()]
        confidence = probabilities[predicted.item()].item()
        
        return predicted_class, confidence, probabilities.cpu().numpy()
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return None, 0, None

def find_actuator_index(model, actuator_name):
    for i in range(model.nu):
        if model.actuator(i).name == actuator_name:
            return i
    return -1

def get_finger_mapping():
    finger_map = {
        'thumb': ['rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1'],
        'index': ['rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0'],  # 검지
        'middle': ['rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0'],  # 중지
        'ring': ['rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0'],    # 약지
        'little': ['rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0'],  # 소지
        'wrist': ['rh_A_WRJ2', 'rh_A_WRJ1']  # 손목
    }
    return finger_map

def set_lfj5_to_zero(model, data):
    lfj5_idx = find_actuator_index(model, 'rh_A_LFJ5')
    
    if lfj5_idx >= 0:
        data.ctrl[lfj5_idx] = 0.0

def make_fist(model, data):
    """주먹 쥐기 동작"""
    print("주먹 쥐기 동작 실행")
    
    for i in range(model.nu):
        data.ctrl[i] = 0.0
    
    finger_map = get_finger_mapping()
    
    for finger, joints in finger_map.items():
        if finger == 'wrist':
            for joint in joints:
                idx = find_actuator_index(model, joint)
                if idx >= 0:
                    data.ctrl[idx] = 0.0
        else:
            for joint in joints:
                if joint != 'rh_A_LFJ5':
                    idx = find_actuator_index(model, joint)
                    if idx >= 0:
                        if 'THJ1' in joint:
                            data.ctrl[idx] = 1.5
                        elif 'THJ2' in joint:
                            data.ctrl[idx] = 0.65
                        elif 'THJ3' in joint:
                            data.ctrl[idx] = -0.0775
                        elif 'THJ4' in joint:
                            data.ctrl[idx] = 0.3
                        elif 'THJ5' in joint:
                            data.ctrl[idx] = -0.85
                        elif 'J4' in joint:
                            data.ctrl[idx] = 0
                        else:
                            data.ctrl[idx] = 1.5    
    set_lfj5_to_zero(model, data)

def open_hand(model, data):
    """손 펴기 (보자기) 동작"""
    print("손 펴기 동작 실행")
    
    for i in range(model.nu):
        data.ctrl[i] = 0.0

def fold_index(model, data):
    '''엄지만 굽히는 동작'''
    print("엄지 굽히기")
    open_hand(model, data)

    finger_map = get_finger_mapping()

    for joint in finger_map['index']:
        idx = find_actuator_index(model, joint)
        if "J3" in joint:
            data.ctrl[idx] = 1.0
        elif "J0" in joint:
            data.ctrl[idx] = 2.4

def fold_mid(model, data):
    '''중지만 굽히는 동작'''
    print("중지 굽히기")
    open_hand(model, data)

    finger_map = get_finger_mapping()

    for joint in finger_map['middle']:
        idx = find_actuator_index(model, joint)
        if "J3" in joint:
            data.ctrl[idx] = 1.0
        elif "J0" in joint:
            data.ctrl[idx] = 2.4

def fold_ring(model, data):
    '''약지만 굽히는 동작'''
    print("약지 굽히기")
    open_hand(model, data)

    finger_map = get_finger_mapping()

    for joint in finger_map['ring']:
        idx = find_actuator_index(model, joint)
        if "J3" in joint:
            data.ctrl[idx] = 1.0
        elif "J0" in joint:
            data.ctrl[idx] = 2.4

def fold_pinky(model, data):
    '''소지만 굽히는 동작'''
    print("소지 굽히기")
    open_hand(model, data)

    finger_map = get_finger_mapping()

    for joint in finger_map['little']:
        idx = find_actuator_index(model, joint)
        if "J3" in joint:
            data.ctrl[idx] = 1.0
        elif "J0" in joint:
            data.ctrl[idx] = 2.4

def check_key_press():
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8', errors='ignore')
        return key
    return None

def execute_gesture(class_name, mj_model, mj_data):
    if class_name == 'fist':
        make_fist(mj_model, mj_data)
    elif class_name == 'open':
        open_hand(mj_model, mj_data)
    elif class_name == 'index':
        fold_index(mj_model, mj_data)
    elif class_name == 'mid':
        fold_mid(mj_model, mj_data)
    elif class_name == 'ring':
        fold_ring(mj_model, mj_data)
    elif class_name == 'pinky':
        fold_pinky(mj_model, mj_data)
    else:
        print(f"알 수 없는 클래스: {class_name}")

def get_buffer_image(buffer_dir, classification_model, last_processed_file=None):
    """Buffer 폴더에서 가장 최신 이미지를 가져와서 처리"""
    
    # Buffer 폴더 존재 확인
    if not os.path.exists(buffer_dir):
        return None, last_processed_file
    
    # PNG 파일 목록 가져오기
    png_files = glob.glob(os.path.join(buffer_dir, "*.png"))
    if not png_files:
        return None, last_processed_file
    
    # 가장 최신 파일 선택
    latest_file = max(png_files, key=os.path.getctime)
    
    # 이전에 처리한 파일과 같다면 스킵
    if latest_file == last_processed_file:
        return None, last_processed_file
    
    # 파일이 완전히 쓰여졌는지 확인
    if not is_file_ready(latest_file):
        return None, last_processed_file
    
    print(f"새 이미지 감지: {latest_file}")
    
    # 이미지 예측 수행
    predicted_class, confidence, probabilities = predict_gesture(
        classification_model, latest_file)
    
    if predicted_class:
        print(f"예측 결과: {predicted_class} (신뢰도: {confidence:.4f})")
        
        # 처리 완료 후 파일 삭제
        os.remove(latest_file)
        print(f"파일 삭제 완료: {latest_file}")
        
        return predicted_class, latest_file
    else:
        return None, last_processed_file

def is_file_ready(file_path, max_wait=0.05):  # 더 짧게
    """파일이 완전히 쓰여졌는지 확인"""
    try:
        initial_size = os.path.getsize(file_path)
        time.sleep(max_wait)
        current_size = os.path.getsize(file_path)
        return initial_size == current_size
    except:
        return False

def clear_buffer_folder(buffer_dir):
    """Buffer 폴더의 모든 이미지 파일 삭제"""
    png_files = glob.glob(os.path.join(buffer_dir, "*.png"))
    for file in png_files:
        os.remove(file)
    print(f"Buffer 폴더 정리 완료: {len(png_files)}개 파일 삭제")

# 메인 함수
def main():
    
    print("손 제스처 인식 및 MuJoCo 제어 프로그램 시작...")
    print("키 입력 설명:")
    print("n: RANDOM IMAGE PREDICTION")
    print("b: BUFFER IMAGE PREDICTION")      
    print("a: AUTO MONITORING ON/OFF")         
    print("c: CLEAR BUFFER FOLDER")
    print("i: 모델 정보 출력")
    print("q: 종료")
    
    model_type = 'mobilenet'
    MODEL_PATH = V10_MODEL_PATH
    # MODEL_PATH = VERA_MODEL_PATH
    classification_model = load_model(MODEL_PATH, model_type)
    
    if classification_model is None:
        print("분류 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    mj_model, mj_data = load_mujoco_model(HAND_MODEL_PATH)
    
    if mj_model is None or mj_data is None:
        print("MuJoCo 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    mujoco.mj_resetData(mj_model, mj_data)

    buffer_dir = r"C:\Users\verasonics\Desktop\Buffer"

    # 자동 모니터링 관련 변수
    auto_monitoring = False
    last_processed_file = None
    frame_count = 0  # 프레임 카운터
    
    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            open_hand(mj_model, mj_data)
            
            last_prediction = None
            running = True
            
            while running:
                key = check_key_press()
                
                if key:
                    if key == 'n':
                        img_path, true_class = get_random_image()
                        if img_path:
                            print(f"\n랜덤 이미지 선택: {img_path}")
                            print(f"실제 클래스: {true_class}")
                            
                            predicted_class, confidence, probabilities = \
                                predict_gesture(classification_model, img_path)
                            
                            if predicted_class:
                                print(f"예측 클래스: {predicted_class} (신뢰도: {confidence:.4f})")
                                execute_gesture(predicted_class, mj_model, mj_data)
                                last_prediction = predicted_class
                        else:
                            print("이미지를 찾을 수 없습니다.")

                    elif key == 'b':
                        # 수동으로 Buffer 이미지 체크
                        print("\nBuffer 폴더에서 이미지 확인 중...")
                        predicted_class, last_processed_file = get_buffer_image(
                            buffer_dir, classification_model, last_processed_file)
                        
                        if predicted_class:
                            execute_gesture(predicted_class, mj_model, mj_data)
                            last_prediction = predicted_class
                        else:
                            print("새로운 이미지가 없습니다.")
                    
                    elif key == 'a':
                        # 자동 모니터링 토글
                        auto_monitoring = not auto_monitoring
                        if auto_monitoring:
                            print("자동 모니터링 시작")
                        else:
                            print("자동 모니터링 중지")
                    
                    elif key == 'c':
                        clear_buffer_folder(buffer_dir)
                    
                    elif key == 'q':
                        print("프로그램을 종료합니다...")
                        running = False

                if auto_monitoring:
                    # 매 10프레임마다 체크 (60Hz에서 6Hz로 체크)
                    if frame_count % 5 == 0:
                        predicted_class, last_processed_file = get_buffer_image(
                            buffer_dir, classification_model, last_processed_file)
                        
                        if predicted_class:
                            execute_gesture(predicted_class, mj_model, mj_data)
                            last_prediction = predicted_class
                
                # 모든 프레임에서 rh_A_LFJ5 액추에이터 값을 0으로 유지
                set_lfj5_to_zero(mj_model, mj_data)
                
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                time.sleep(1/60)
    
    except Exception as e:
        print(f"오류 발생: {e}")
    
    finally:
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()