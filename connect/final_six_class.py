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

DATA_ROOT = "./cat_gest" 
MODEL_PATH = "./best_hand_gesture_mobilenet_model.pth"
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

# 메인 함수
def main():
    
    print("손 제스처 인식 및 MuJoCo 제어 프로그램 시작...")
    print("키 입력 설명:")
    print("n: RANDOM IMAGE PREDICTION")
    print("0: OPEN")
    print("1: INDEX")
    print("2: MIDDLE")
    print("3: RING")
    print("4: PINKY")
    print("5: FIST")
    print("i: 모델 정보 출력")
    print("q: 종료")
    
    model_type = 'mobilenet'
    classification_model = load_model(MODEL_PATH, model_type)
    
    if classification_model is None:
        print("분류 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    mj_model, mj_data = load_mujoco_model(HAND_MODEL_PATH)
    
    if mj_model is None or mj_data is None:
        print("MuJoCo 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    mujoco.mj_resetData(mj_model, mj_data)
    
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
                    
                    elif key == '0':
                        open_hand(mj_model, mj_data)
                        last_prediction = 'open'
                    elif key == '1':
                        fold_index(mj_model, mj_data)
                        last_prediction = 'index'
                    elif key == '2':
                        fold_mid(mj_model, mj_data)
                        last_prediction = 'mid'
                    elif key == '3':
                        fold_ring(mj_model, mj_data)
                        last_prediction = 'ring'
                    elif key == '4':
                        fold_pinky(mj_model, mj_data)
                        last_prediction = 'pinky'
                    elif key == '5':
                        make_fist(mj_model, mj_data)
                        last_prediction = 'fist'
                    elif key == 'q':
                        print("프로그램을 종료합니다...")
                        running = False
                
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