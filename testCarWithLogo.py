import torch
import clip
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from natsort import natsorted

#設定好使用的model以及CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# 圖片資料夾路徑設定
folder_path = '/home/yangl/CLIP/CarWithLogo'
#folder_path = '/home/yangl/CLIP/CarPhoto'

#讀取資料夾中圖片
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
img_files = natsorted(img_files)  # 使用自然排序
#將Token設成一個陣列
token = np.array(["a car with NEXT logo", "a car with WATER & COFFEE logo", "a car with SCHOOL BUS logo", "a car with 217 logo", "a green truck","a trash truck","a car with FedEx logo","a car with coca cola logo","a car with pepsi logo","a car with 7up logo","a car with DHL logo", "a car with Prime logo","a car with American Lager logo"])
#token = np.array(["a BMW car", "a Benz car", "a Ford car", "a Audi car", "a Ferrari car","a Honda car","a Lamborgini car","a Lexus car","a Mazda car","a Porcshe car"])


for image in img_files:
    print(f'正在處理圖片: {image}')
    image = preprocess(Image.open(os.path.join(folder_path, image))).unsqueeze(0).to(device)
    text = clip.tokenize(token).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # 將二維數組轉換為一維
    probs = np.array(probs).flatten()
    #print(probs)
    maxProb = max(probs)
    maxProb_index = np.argmax(probs)
    print("預測的結果為:",token[maxProb_index], "機率=",maxProb)  # prints: [[0.9927937  0.00421068 0.00299572]]
    # # 將二維數組轉換為一維
    # probs = np.array(probs).flatten()
    # # 創建柱狀圖
    # plt.bar(range(len(probs)), probs)
    # plt.title('graph')
    # plt.xlabel('label')
    # plt.ylabel('prob')
    # plt.show()







