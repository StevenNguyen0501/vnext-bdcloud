import base64
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2
# from ultralytics import YOLO
# from typing import Dict
import os
import pandas as pd
from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8
import boto3
from datetime import datetime
from http.client import HTTPException
from dotenv import load_dotenv, dotenv_values
import json

load_dotenv()
app = FastAPI()
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
class ImageData(BaseModel):
    imageBase64: str
    petName: str
    userName: str
    userEmail: str
    userPhoneNumber: str

def visualize_result(results, img):
    for result in results:
        r = result["bbox"]
        class_id = int(result["class_id"])
        class_name = class_names[class_id]
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, class_name, (r[2] + 5, r[1] + 20), font, 0.5, (0, 238, 238), 1, cv2.LINE_AA)
    return img

def process_image_with_model(img_path):
    model = YOLOv8('last.onnx')
    # infer result
    img = Image.open(img_path)
    results = model(img, size=640, conf_thres=0.3, iou_thres=0.5)
    image = cv2.imread(img_path)
    img_visualized = visualize_result(results, image)
    # cv2.imwrite('img_visualized.jpg', img_visualized)

    # Processing for color extraction
    xywh = [list(result["bbox"]) for result in results]
    xy = [[int(c[0]), int(c[1])] for c in xywh]
    xy = list(xy)
    cls = [int(result["class_id"]) for result in results]
    result_dict = {}
    for cl, xy_val in zip(cls, xy):
        result_dict[lst_classes[int(cl)]] = xy_val

    urine_colors = {}
    for name, xy in result_dict.items():
        x, y = map(int, xy)
        rgb_value = image[y, x].tolist()
        urine_colors[name] = rgb_value

    print(urine_colors)

    return img_visualized, urine_colors

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
    # List of variable names
    variable_names = [
        "LEUKOCYTES", "NITRITE", "UROBILINOGEN", "PROTEIN", "PH",
        "BLOOD", "GRAVITY", "KETONE", "BILIRUBIN", "GLUCOSE"
    ]
    # Deserialize environment variables into dictionaries
    variables = {name: json.loads(os.getenv(name)) for name in variable_names}
    # Unpack variables for individual use (optional)
    LEUKOCYTES, NITRITE, UROBILINOGEN, PROTEIN, PH, BLOOD, GRAVITY, KETONE, BILIRUBIN, GLUCOSE = variables.values()
    for index in test_indices:
        urine_color = urine_colors[index]
        if index == "Leukocytes":
            result[index] = find_closest_color_cie(urine_color, LEUKOCYTES)
        elif index == "Nitrite":
            result[index] = find_closest_color_cie(urine_color, NITRITE)
        elif index == "Urobilinogen":
            result[index] = find_closest_color_cie(urine_color, UROBILINOGEN)
        elif index == "Protein":
            result[index] = find_closest_color_cie(urine_color, PROTEIN)
        elif index == "pH":
            result[index] = find_closest_color_cie(urine_color, PH)
        elif index == "Blood":
            result[index] = find_closest_color_cie(urine_color, BLOOD)
        elif index == "Specific":
            result[index] = find_closest_color_cie(urine_color, GRAVITY)
        elif index == "Ketone":
            result[index] = find_closest_color_cie(urine_color, KETONE)
        elif index == "Bilirubin":
            result[index] = find_closest_color_cie(urine_color, BILIRUBIN)
        elif index == "Glucose":
            result[index] = find_closest_color_cie(urine_color, GLUCOSE)
    return result
def analyze_urine_test_svm(urine_colors):
    result = {}

    def find_closest_color_svm(target, file_path, rgb_value):
        datasets = pd.read_csv(file_path)
        X = datasets.iloc[:, [0,1,2]].values
        Y = datasets.iloc[:, 3].values

        # Splitting the dataset into the Training set and Test set

        from sklearn.model_selection import train_test_split
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

        # Feature Scaling

        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_Train = sc_X.fit_transform(X_Train)
        X_Test = sc_X.transform(X_Test)
        # Fitting the classifier into the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        # print(X_Train)
        # print(Y_Train)
        classifier.fit(X_Train, Y_Train)

        # Predicting the test set results

        Y_Pred = classifier.predict(X_Test)

        # Making the Confusion Matrix

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_Test, Y_Pred)

        # Chuẩn bị input mới
        new_input = [rgb_value]  # Tạo input mới từ dữ liệu bạn cung cấp

        # Tiêu chuẩn hóa input mới
        new_input_scaled = sc_X.transform(new_input)

        # Dự đoán kết quả cho input mới
        predicted_class = classifier.predict(new_input_scaled)
        return predicted_class[0]  #closest_color

    test_indices = ['Bilirubin','Blood','Glucose','Ketone','Leukocytes','Nitrite','Protein','Specific','Urobilinogen','pH']
    for index in test_indices:
        urine_color = urine_colors[index]
        if index == "Leukocytes":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Nitrite":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Urobilinogen":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Protein":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "pH":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Blood":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Specific":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Ketone":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Bilirubin":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)
        elif index == "Glucose":
            result[index] = find_closest_color_svm(urine_color, f"C:/Users/thinhnp/PycharmProjects/fastAPI_Urine/data_SVM/{index}.csv", urine_color)

    return result
def connect_to_dynamodb():
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    dynamodb = boto3.resource('dynamodb',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=aws_region)

    # Chọn bảng DynamoDB để làm việc
    table = dynamodb.Table('dev-db-buddycloud-detailresults')
    return dynamodb, table
@app.post("/process_image_Ciede2000/")
async def process_image(image_data: ImageData):
    base64_data = image_data.imageBase64
    binary_data = base64.b64decode(base64_data)
    temp_image_path = "file_image_tmp.jpg"
    image = Image.open(io.BytesIO(binary_data))
    image.save(temp_image_path)
    # Lấy thời gian hiện tại và định dạng theo đúng định dạng 'YYYY-MM-DD HH:MM:SS'
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    safe_datetime = current_datetime.replace(':', '_').replace(' ', '_')
    # Load hình ảnh từ đường dẫn tệp tin
    image1 = Image.open(temp_image_path)
    # Đường dẫn thư mục để lưu hình ảnh
    folder_path = 'images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_filename = rf'{safe_datetime}.png'
    saved_image_path = os.path.join(folder_path, image_filename)
    print('saved_image_path', saved_image_path)
    # Lưu hình ảnh
    image1.save(saved_image_path, format="PNG")

    # Xử lý hình ảnh với mô hình và trả về màu của nước tiểu
    img_visualized, urine_colors = process_image_with_model(temp_image_path)
    result = analyze_urine_test_cie(urine_colors)
    image = img_visualized
    # Đường dẫn thư mục để lưu hình ảnh
    folder1_path = 'detectedImageURL'
    if not os.path.exists(folder1_path):
        os.makedirs(folder1_path)
    image_filename = rf'{safe_datetime}.png'
    saved_image_path1 = os.path.join(folder1_path, image_filename)
    # Lưu hình ảnh
    cv2.imwrite(saved_image_path1,image)
    # Kiểm tra sự tồn tại của userName
    # if check_username(image_data.userName):
    #     return {"error": "Tên người dùng đã tồn tại. Vui lòng chọn một tên người dùng khác."}
    dynamodb, table = connect_to_dynamodb()
    # Thực hiện phép quét (scan) để lấy số lượng mục trong bảng
    response = table.scan(Select='COUNT')
    # Lấy số lượng mục hiện có trong bảng
    item_count = response['Count']
    # Nếu không có mục nào trong bảng, ID_counter sẽ được khởi tạo là 1, ngược lại sẽ là số lượng mục hiện có + 1
    ID_counter = 1 if item_count == 0 else item_count + 1
    userID = str(ID_counter) + image_data.userName
    # Thêm mục mới vào bảng
    table.put_item(
        Item={
            "ID": str(ID_counter),
            "userID": str(userID),
            "userName": image_data.userName,
            "petName": image_data.petName,
            "petID": str(ID_counter),
            "userPhoneNumber": image_data.userPhoneNumber,
            "userEmail": image_data.userEmail,
            "timestamp": current_datetime,
            "resultsID": str(ID_counter),
            "resultsDetail": result,
            "rawImageURL": saved_image_path,
            "detectedImageURL": saved_image_path1
        }
    )
    ID_counter +=1
    # Xóa hình ảnh tạm thời sau khi đã xử lý
    #os.remove(temp_image_path)
    # Tạo hình ảnh từ mảng Numpy
    cv2_rgb = cv2.cvtColor(img_visualized, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_rgb)
    # Tạo chuỗi base64 từ hình ảnh
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    result['base_64']=base64_image
    return result
@app.post("/process_image_SVM/")
async def process_image(image_data: ImageData):
    base64_data = image_data.imageBase64
    binary_data = base64.b64decode(base64_data)
    temp_image_path = "file_image_tmp.jpg"
    image = Image.open(io.BytesIO(binary_data))
    image.save(temp_image_path)
    # Lấy thời gian hiện tại và định dạng theo đúng định dạng 'YYYY-MM-DD HH:MM:SS'
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    safe_datetime = current_datetime.replace(':', '_').replace(' ', '_')
    # Load hình ảnh từ đường dẫn tệp tin
    image1 = Image.open(temp_image_path)
    # Đường dẫn thư mục để lưu hình ảnh
    folder_path = 'images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_filename = rf'{safe_datetime}.png'
    saved_image_path = os.path.join(folder_path, image_filename)
    print('saved_image_path', saved_image_path)
    # Lưu hình ảnh
    image1.save(saved_image_path, format="PNG")

    # Xử lý hình ảnh với mô hình và trả về màu của nước tiểu
    img_visualized, urine_colors = process_image_with_model(temp_image_path)
    result = analyze_urine_test_svm(urine_colors)
    image = img_visualized
    # Đường dẫn thư mục để lưu hình ảnh
    folder1_path = 'detectedImageURL'
    if not os.path.exists(folder1_path):
        os.makedirs(folder1_path)
    image_filename = rf'{safe_datetime}.png'
    saved_image_path1 = os.path.join(folder1_path, image_filename)
    # Lưu hình ảnh
    cv2.imwrite(saved_image_path1,image)
    # Kiểm tra sự tồn tại của userName
    # if check_username(image_data.userName):
    #     return {"error": "Tên người dùng đã tồn tại. Vui lòng chọn một tên người dùng khác."}
    dynamodb, table = connect_to_dynamodb()
    # Thực hiện phép quét (scan) để lấy số lượng mục trong bảng
    response = table.scan(Select='COUNT')
    # Lấy số lượng mục hiện có trong bảng
    item_count = response['Count']
    # Nếu không có mục nào trong bảng, ID_counter sẽ được khởi tạo là 1, ngược lại sẽ là số lượng mục hiện có + 1
    ID_counter = 1 if item_count == 0 else item_count + 1
    userID = str(ID_counter) + image_data.userName
    # Thêm mục mới vào bảng
    table.put_item(
        Item={
            "ID": str(ID_counter),
            "userID": str(userID),
            "userName": image_data.userName,
            "petName": image_data.petName,
            "petID": str(ID_counter),
            "userPhoneNumber": image_data.userPhoneNumber,
            "userEmail": image_data.userEmail,
            "timestamp": current_datetime,
            "resultsID": str(ID_counter),
            "resultsDetail": result,
            "rawImageURL": saved_image_path,
            "detectedImageURL": saved_image_path1
        }
    )
    ID_counter +=1
    # Tạo hình ảnh từ mảng Numpy
    cv2_rgb = cv2.cvtColor(img_visualized, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_rgb)
    # Tạo chuỗi base64 từ hình ảnh
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    result['base_64']=base64_image
    return result

@app.post("/item/{id}/{timestamp}")
async def read_item(id: str, timestamp: str):
    try:
        dynamodb, table = connect_to_dynamodb()
        # Truy vấn dữ liệu từ bảng
        response = table.get_item(
            Key={
                'ID': id,
                'timestamp': timestamp
            }
        )
        item = response['Item']  # Thay đổi ở đây

        # Kiểm tra xem item có tồn tại hay không
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_dynamodb_item")
async def update_dynamodb_item(item_id: str, timestamp: str, userPhoneNumber: str, detectedImageURL: str, petID: str, petName: str, rawImageURL: str, userID: str, resultsDetail: str, userEmail: str, userName: str, resultsID: str):

    # Cập nhật mục trong bảng
    try:
        dynamodb, table = connect_to_dynamodb()
        response = table.update_item(
            Key={
                'ID': item_id,
                'timestamp': timestamp
            },
            UpdateExpression='SET userPhoneNumber = :val1, detectedImageURL = :val2, petName = :val3, userEmail = :val4, userName = :val5, petID = :val6, rawImageURL = :val7, userID = :val8, resultsDetail = :val9, resultsID = :val10',
            ExpressionAttributeValues={
                ':val1': userPhoneNumber,
                ':val2': detectedImageURL,
                ':val3': petName,
                ':val4': userEmail,
                ':val5': userName,
                ':val6': petID,
                ':val7': rawImageURL,
                ':val8': userID,
                ':val9': resultsDetail,
                ':val10': resultsID
            }
        )
        return {"message": "Cập nhật thành công!", "response": response}
    except Exception as e:
        return {"error": f"Lỗi khi cập nhật mục trong bảng: {e}"}
@app.post("/items/{item_id}")
async def delete_item(item_id: str, timestamp: str):
    try:
        dynamodb, table = connect_to_dynamodb()
        response = table.delete_item(
            Key={
                'ID': item_id,
                'timestamp': timestamp
            }
        )
        return {"message": "Item deleted successfully", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))