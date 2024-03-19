import base64
import io

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2
from ultralytics import YOLO
from typing import Dict
import os
import tempfile
import pandas as pd
from PIL import Image

app = FastAPI()
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

def process_image_with_model(img_path):
    model = YOLO('last.pt')
    results = model.predict(source=img_path, save=True, save_txt=True)  # save plotted images and txt

    def visualize_result(results, img_path):
        img = cv2.imread(img_path)  # Đọc hình ảnh để vẽ lên đó
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Get boxes in numpy format
            for box in boxes:
                r = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
                text_x = r[2] + 5  # Adjust the x-coordinate to the right of the box
                text_y = r[1] + text_size[1] + 5  # Adjust the y-coordinate for better positioning
                cv2.putText(img, class_name, (text_x, text_y), font, font_scale, (0, 238, 238), font_thickness, cv2.LINE_AA)
        return img

    img_visualized = visualize_result(results, img_path)
    # Processing for color extraction
    xywh = [r.boxes.xywh for r in results]
    xy = [c[:, :2] for c in xywh]
    cls = [r.boxes.cls for r in results]
    names_list = ['Bilirubin','Blood','Glucose','Ketone','Leukocytes','Nitrite','Protein','Specific','Urobilinogen','pH']
    result_dict = {}
    for cl, xy_val in zip(cls, xy):
        for c, xy_single in zip(cl.int(), xy_val):
            result_dict[names_list[int(c)]] = xy_single.tolist()

    image = cv2.imread(img_path)
    urine_colors = {}
    for name, xy in result_dict.items():
        x, y = map(int, xy)
        rgb_value = image[y, x].tolist()
        urine_colors[name] = rgb_value

    print(urine_colors)

    return img_visualized, urine_colors

def analyze_urine_test_svm(urine_colors):
    result = {}

    def find_closest_color_svm(target, file_path, rgb_value):
        # Importing the datasets
        print('aaa', file_path)
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

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Tạo tệp tạm thời và ghi nội dung hình ảnh vào đó
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image_path = temp_image.name
        temp_image.write(contents)

    # Xử lý hình ảnh với mô hình và trả về màu của nước tiểu
    img_visualized, urine_colors = process_image_with_model(temp_image_path)
    image_pil = Image.fromarray(img_visualized)  # Chuyển đổi numpy array thành đối tượng Image
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Encode bytes as base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Now you have the base64 encoded string representation of the image
    #print(base64_image)


    result = analyze_urine_test_svm(urine_colors)

    # Xóa hình ảnh tạm thời sau khi đã xử lý
    os.remove(temp_image_path)
    responce = {"img_base64":base64_image}
    result["img_base64"] = base64_image
    response = result
    return response
