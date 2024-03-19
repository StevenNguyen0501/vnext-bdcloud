from io import BytesIO

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import cv2
from torch.mps import event
# from ultralytics import YOLO
# import os
# import pandas as pd
from pydantic import BaseModel

import base64
from datetime import datetime
import tempfile
import os
import boto3

from yolo_onnx.yolov8_onnx import YOLOv8
import json
import base64
from io import BytesIO

class_names = {
    0: 'Bilirubin',
    1: 'Blood',
    2: 'Glucose',
    3: 'Ketone',
    4: 'Leukocytes',
    5: 'Nitrite',
    6: 'Protein',
    7: 'Specific',
    8: 'Urobilinogen',
    9: 'pH'
}

lst_classes = list(class_names.values())

app = FastAPI()
class ImageData(BaseModel):
    imageBase64: str
    petName : str
    userName :str
    userEmail : str
    userPhoneNumber : str

class UrineColor(BaseModel):
    Bilirubin: list
    Blood: list
    Glucose: list
    Ketone: list
    Leukocytes: list
    Nitrite: list
    Protein: list
    Specific: list
    Urobilinogen: list
    pH: list

def visualize_result(results, img):
    for result in results:
        r = result["bbox"]
        class_id = int(result["class_id"])
        class_name = class_names[class_id]
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, class_name, (r[2] + 5, r[1] + 20), font, 0.5, (0, 238, 238), 1, cv2.LINE_AA)
    return img

def process_image_with_model1(img_path):
  
    model = YOLOv8('weight/last.onnx')
    print("Model YOLOv8: ", model)

    app = FastAPI()
    # infer result
    img = Image.open(img_path)
    results = model(img, size=640, conf_thres=0.3, iou_thres=0.5)
    print("Results YOLOv8: ", results)

    image = cv2.imread(img_path)
    img_visualized = visualize_result(results, image)
    # cv2.imwrite('img_visualized.jpg', img_visualized)

    # Processing for color extraction
    xywh = [list(result["bbox"]) for result in results]
    print('xywh: ', xywh)
    print('type xywh: ', type(xywh))
    xy = [[int(c[0]), int(c[1])] for c in xywh]
    xy = list(xy)
    print("xy: ", xy)
    cls = [int(result["class_id"]) for result in results]
    print("cls: ", cls)

    result_dict = {}
    for cl, xy_val in zip(cls, xy):
        print(cl)
        print(xy_val)
        result_dict[lst_classes[int(cl)]] = xy_val

    urine_colors = {}
    for name, xy in result_dict.items():
        x, y = map(int, xy)
        rgb_value = image[y, x].tolist()
        urine_colors[name] = rgb_value

    print(urine_colors)

    return img_visualized, urine_colors

def analyze_urine_test(urine_colors):
    result = {}

    def find_closest_color(target, color_dict):
        min_distance = float('inf')
        closest_color = None

        for key, value in color_dict.items():
            print(target, value)
            distance = sum((a - b) ** 2 for a, b in zip(target, value))
            if distance < min_distance:
                min_distance = distance
                closest_color = key

        return closest_color

    for index in lst_classes:
        urine_color = urine_colors[index]
        if index == "Leukocytes":
            result[index] = find_closest_color(urine_color, {"NEGATIVE": [254, 248, 188],
                                                             "TRACE(15)": [229, 218, 174],
                                                             "SMALL(75)": [206, 157, 149],
                                                             "MODERATE(125)": [166, 120, 153],
                                                             "LARGE(500)": [110, 83, 124]})
        elif index == "Nitrite":
            result[index] = find_closest_color(urine_color, {"NEGATIVE": [253, 250, 222],
                                                             "POSITIVE1": [251, 220, 218],
                                                             "POSITIVE2": [247, 181, 200],
                                                             "POSITIVE3": [238, 78, 130]})
        elif index == "Urobilinogen":
            result[index] = find_closest_color(urine_color, {"NORMAL(32)": [254, 211, 174],
                                                             "NORMAL(16)": [248, 168, 133],
                                                             "32": [243, 131, 140],
                                                             "64": [230, 111, 128],
                                                             "128": [230, 78, 130]})
        elif index == "Protein":
            result[index] = find_closest_color(urine_color, {"NEGATIVE": [222, 229, 125],
                                                             "TRACE": [187, 215, 106],
                                                             "0.3": [172, 212, 130],
                                                             "1.0": [119, 189, 151],
                                                             "3.0": [94, 178, 169],
                                                             ">=20": [0, 148, 149]})
        elif index == "pH":
            result[index] = find_closest_color(urine_color, {"5.0": [245, 139, 79],
                                                             "6.0": [249, 165, 85],
                                                             "6.5": [253, 195, 109],
                                                             "7.0": [208, 189, 98],
                                                             "7.5": [136, 148, 85],
                                                             "8.0": [86, 173, 145],
                                                             "8.5": [0, 127, 129]})
        elif index == "Blood":
            result[index] = find_closest_color(urine_color, {"NEGATIVE": [250, 174, 76],
                                                             "TRACE(NON-HEMOLYZED)": [250, 284, 77],
                                                             "TRACE(10)": [207, 161, 65],
                                                             "SMALL(25)": [161, 156, 84],
                                                             "MODERATE(80)": [116, 156, 122],
                                                             "LARGE(200)": [69, 128, 108]})
        elif index == "Gravity":
            result[index] = find_closest_color(urine_color, {"1.000": [2, 113, 126],
                                                             "1.005": [76, 117, 102],
                                                             "1.010": [123, 136, 105],
                                                             "1.015": [155, 141, 58],
                                                             "1.020": [175, 161, 52],
                                                             "1.025": [197, 167, 48],
                                                             "1.030": [210, 171, 43]})
        elif index == "Ketone":
            result[index] = find_closest_color(urine_color, {"NEGATIVE": [251, 188, 149],
                                                             "TRACE(0.5)": [246, 158, 137],
                                                             "SMALL(1.5)": [243, 131, 140],
                                                             "MODERATE(4.0)": [201, 88, 116],
                                                             "LARGE1(8.0)": [150, 58, 102],
                                                             "LARGE2(16.0)": [120, 41, 90]})
        elif index == "Bilirubin":
            result[index] = find_closest_color(urine_color, {"NEGATIVE1": [253, 250, 222],
                                                             "NEGATIVE2": [253, 223, 144],
                                                             "SMALL(17)": [251, 187, 131],
                                                             "MODERATE(50)": [208, 146, 136],
                                                             "LARGE(100)": [171, 127, 131]})
        elif index == "Glucose":
            result[index] = find_closest_color(urine_color, {"NEGATIVE1": [111, 203, 220],
                                                             "NEGATIVE2": [141, 208, 187],
                                                             "TRACE(5)": [152, 207, 148],
                                                             "15": [139, 171, 106],
                                                             "30": [164, 129, 67],
                                                             "60": [157, 105, 37],
                                                             "110": [136, 89, 41]})

    return result
def rgb2lab(inputColor):

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab
import math

def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625 # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)

    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi
    else: h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))

    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00
def analyze_urine_test_cie(urine_colors):
    result = {}

    def find_closest_color_cie(target, color_dict):
      min_distance = float('inf')
      closest_color = None

      for key, value in color_dict.items():
          distance = CIEDE2000(rgb2lab(target), rgb2lab(value)) #sum((a - b) ** 2 for a, b in zip(target, value))
          if distance < min_distance:
              min_distance = distance
              closest_color = key

      return closest_color

    test_indices = ['Bilirubin','Blood','Glucose','Ketone','Leukocytes','Nitrite','Protein','Specific','Urobilinogen','pH']

    for index in test_indices:
        urine_color = urine_colors[index]
        if index == "Leukocytes":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE": [254, 248, 188],
                                                             "TRACE(15)": [229, 218, 174],
                                                             "SMALL(75)": [206, 157, 149],
                                                             "MODERATE(125)": [166, 120, 153],
                                                             "LARGE(500)": [110, 83, 124]})
        elif index == "Nitrite":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE": [253, 250, 222],
                                                             "POSITIVE1": [251, 220, 218],
                                                             "POSITIVE2": [247, 181, 200],
                                                             "POSITIVE3": [238, 78, 130]})
        elif index == "Urobilinogen":
            result[index] = find_closest_color_cie(urine_color, {"NORMAL(32)": [254, 211, 174],
                                                             "NORMAL(16)": [248, 168, 133],
                                                             "32": [243, 131, 140],
                                                             "64": [230, 111, 128],
                                                             "128": [230, 78, 130]})
        elif index == "Protein":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE": [222, 229, 125],
                                                             "TRACE": [187, 215, 106],
                                                             "0.3": [172, 212, 130],
                                                             "1.0": [119, 189, 151],
                                                             "3.0": [94, 178, 169],
                                                             ">=20": [0, 148, 149]})
        elif index == "pH":
            result[index] = find_closest_color_cie(urine_color, {"5.0": [245, 139, 79],
                                                             "6.0": [249, 165, 85],
                                                             "6.5": [253, 195, 109],
                                                             "7.0": [208, 189, 98],
                                                             "7.5": [136, 148, 85],
                                                             "8.0": [86, 173, 145],
                                                             "8.5": [0, 127, 129]})
        elif index == "Blood":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE": [250, 174, 76],
                                                             "TRACE(NON-HEMOLYZED)": [250, 284, 77],
                                                             "TRACE(10)": [207, 161, 65],
                                                             "SMALL(25)": [161, 156, 84],
                                                             "MODERATE(80)": [116, 156, 122],
                                                             "LARGE(200)": [69, 128, 108]})
        elif index == "Specific":
            result[index] = find_closest_color_cie(urine_color, {"1.000": [2, 113, 126],
                                                             "1.005": [76, 117, 102],
                                                             "1.010": [123, 136, 105],
                                                             "1.015": [155, 141, 58],
                                                             "1.020": [175, 161, 52],
                                                             "1.025": [197, 167, 48],
                                                             "1.030": [210, 171, 43]})
        elif index == "Ketone":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE": [251, 188, 149],
                                                             "TRACE(0.5)": [246, 158, 137],
                                                             "SMALL(1.5)": [243, 131, 140],
                                                             "MODERATE(4.0)": [201, 88, 116],
                                                             "LARGE1(8.0)": [150, 58, 102],
                                                             "LARGE2(16.0)": [120, 41, 90]})
        elif index == "Bilirubin":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE1": [253, 250, 222],
                                                             "NEGATIVE2": [253, 223, 144],
                                                             "SMALL(17)": [251, 187, 131],
                                                             "MODERATE(50)": [208, 146, 136],
                                                             "LARGE(100)": [171, 127, 131]})
        elif index == "Glucose":
            result[index] = find_closest_color_cie(urine_color, {"NEGATIVE1": [111, 203, 220],
                                                             "NEGATIVE2": [141, 208, 187],
                                                             "TRACE(5)": [152, 207, 148],
                                                             "15": [139, 171, 106],
                                                             "30": [164, 129, 67],
                                                             "60": [157, 105, 37],
                                                             "110": [136, 89, 41]})

    return result
# def analyze_urine_test_svm(urine_colors):
#     result = {}
#
#     def find_closest_color_svm(target, file_path, rgb_value):
#         # Importing the datasets
#         print('aaa', file_path)
#         datasets = pd.read_csv(file_path)
#         X = datasets.iloc[:, [0,1,2]].values
#         Y = datasets.iloc[:, 3].values
#
#         # Splitting the dataset into the Training set and Test set
#
#         from sklearn.model_selection import train_test_split
#         X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#
#         # Feature Scaling
#
#         from sklearn.preprocessing import StandardScaler
#         sc_X = StandardScaler()
#         X_Train = sc_X.fit_transform(X_Train)
#         X_Test = sc_X.transform(X_Test)
#
#         # Fitting the classifier into the Training set
#
#         from sklearn.svm import SVC
#         classifier = SVC(kernel = 'linear', random_state = 0)
#         # print(X_Train)
#         # print(Y_Train)
#         classifier.fit(X_Train, Y_Train)
#
#         # Predicting the test set results
#
#         Y_Pred = classifier.predict(X_Test)
#
#         # Making the Confusion Matrix
#
#         from sklearn.metrics import confusion_matrix
#         cm = confusion_matrix(Y_Test, Y_Pred)
#
#         # Chuẩn bị input mới
#         new_input = [rgb_value]  # Tạo input mới từ dữ liệu bạn cung cấp
#
#         # Tiêu chuẩn hóa input mới
#         new_input_scaled = sc_X.transform(new_input)
#
#         # Dự đoán kết quả cho input mới
#         predicted_class = classifier.predict(new_input_scaled)
#         return predicted_class[0]  #closest_color
#
#     test_indices = ['Bilirubin','Blood','Glucose','Ketone','Leukocytes','Nitrite','Protein','Specific','Urobilinogen','pH']
#     for index in test_indices:
#         urine_color = urine_colors[index]
#         if index == "Leukocytes":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Nitrite":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Urobilinogen":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Protein":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "pH":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Blood":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Specific":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Ketone":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Bilirubin":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#         elif index == "Glucose":
#             result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
#
#     return result

# @app.post("/process_image_L2distance/")
# async def process_image(image_data: ImageData):
#     base64_data = image_data.imageBase64.split(",")[1]
#     binary_data = base64.b64decode(base64_data)
#     temp_image_path = "file_image_tmp.jpg"
#     image = Image.open(BytesIO(binary_data))
#     image.save(temp_image_path)
#     # Xử lý hình ảnh với mô hình và trả về màu của nước tiểu
#     img_visualized, urine_colors = process_image_with_model1(temp_image_path)
#     result = analyze_urine_test(urine_colors)
#     print('ssssss', result)
#     # Chuyển đổi ảnh đã được xử lý thành dạng Base64
#     with open(temp_image_path, "rb") as image_file:
#         img1_base64 = base64.b64encode(image_file.read()).decode("utf-8")
#     # Chuyển đổi ảnh đã được xử lý thành dạng Base64
#     _, buffer = cv2.imencode('.jpg', img_visualized)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')
#
#
#     dynamodb = boto3.resource('dynamodb',
#                               aws_access_key_id='AKIA47CRWLV57NUYSDTM',
#                               aws_secret_access_key='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',
#                               region_name='ap-southeast-1')
#
#     # Chọn bảng DynamoDB để làm việc
#     table = dynamodb.Table('dev-db-buddycloud-detailresults')
#
#     # Lấy thời gian hiện tại và định dạng theo đúng định dạng 'YYYY-MM-DD HH:MM:SS'
#     current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#     # Thực hiện phép quét (scan) để lấy số lượng mục trong bảng
#     response = table.scan(Select='COUNT')
#
#     # Lấy số lượng mục hiện có trong bảng
#     item_count = response['Count']
#
#     # Nếu không có mục nào trong bảng, ID_counter sẽ được khởi tạo là 1, ngược lại sẽ là số lượng mục hiện có + 1
#     ID_counter = 1 if item_count == 0 else item_count + 1
#
#     # Thêm mục mới vào bảng
#     table.put_item(
#         Item={
#             "ID": str(ID_counter),
#             "userID": str(ID_counter),
#             "userName": image_data.userName,
#             "petName": image_data.petName,
#             "petID": str(ID_counter),
#             "userPhoneNumber": image_data.userPhoneNumber,
#             "userEmail": image_data.userEmail,
#             "timestamp": current_datetime,
#             "resultsID": str(ID_counter),
#             "resultsDetail": result,
#             "rawImageURL": img1_base64,  # Thay đổi giá trị tại đây
#             "detectedImageURL": img_base64
#         }
#     )
#
#     ID_counter +=1
#     # Xóa hình ảnh tạm thời sau khi đã xử lý
#     os.remove(temp_image_path)
#
#     return result

ALLOWED_EXTENSIONS = [".jpg", ".png", ".jpeg"]
@app.post("/process_image_L2distance/")
async def process_image(userName: str = Form(...),
                        petName: str = Form(...),
                        userPhoneNumber: str = Form(...),
                        userEmail: str = Form(...),
                        file: UploadFile = File(...)):
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() not in ALLOWED_EXTENSIONS:
        return {"error": f"File extension {file_extension} is not allowed."}

    contents = await file.read()

    # Tạo tệp tạm thời và ghi nội dung hình ảnh vào đó
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_image:
        temp_image_path = temp_image.name
        temp_image.write(contents)
    # Xử lý hình ảnh với mô hình và trả về màu của nước tiểu
    img_visualized, urine_colors = process_image_with_model1(temp_image_path)
    result = analyze_urine_test(urine_colors)
    print('ssssss', result)
    # Chuyển đổi ảnh đã được xử lý thành dạng Base64
    with open(temp_image_path, "rb") as image_file:
        img1_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    # Chuyển đổi ảnh đã được xử lý thành dạng Base64
    _, buffer = cv2.imencode('.jpg', img_visualized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')


    dynamodb = boto3.resource('dynamodb',
                              aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                              aws_secret_access_key='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',
                              region_name='ap-southeast-1')

    # Chọn bảng DynamoDB để làm việc
    table = dynamodb.Table('dev-db-buddycloud-detailresults')

    # Lấy thời gian hiện tại và định dạng theo đúng định dạng 'YYYY-MM-DD HH:MM:SS'
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Thực hiện phép quét (scan) để lấy số lượng mục trong bảng
    response = table.scan(Select='COUNT')

    # Lấy số lượng mục hiện có trong bảng
    item_count = response['Count']

    # Nếu không có mục nào trong bảng, ID_counter sẽ được khởi tạo là 1, ngược lại sẽ là số lượng mục hiện có + 1
    ID_counter = 1 if item_count == 0 else item_count + 1

    # Thêm mục mới vào bảng
    table.put_item(
        Item={
            "ID": str(ID_counter),
            "userID": str(ID_counter),
            "userName": userName,
            "petName": petName,
            "petID": str(ID_counter),
            "userPhoneNumber": userPhoneNumber,
            "userEmail": userEmail,
            "timestamp": current_datetime,
            "resultsID": str(ID_counter),
            "resultsDetail": result,
            "rawImageURL": img1_base64,  # Thay đổi giá trị tại đây
            "detectedImageURL": img_base64
        }
    )

    ID_counter +=1
    # Xóa hình ảnh tạm thời sau khi đã xử lý
    os.remove(temp_image_path)

    return result