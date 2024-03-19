import base64
from datetime import datetime
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import requests
import boto3

app = FastAPI()

def convert_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        image_binary = f.read()
        base64_encoded = base64.b64encode(image_binary)
        return base64_encoded.decode('utf-8')

def process_image_and_save_result(image_path: str, userName: str, petName: str, userPhoneNumber: str, userEmail: str, img1_base64: str):
    # Kết nối với DynamoDB
    dynamodb = boto3.resource('dynamodb',
                              aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                              aws_secret_access_key='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',
                              region_name='ap-southeast-1')

    # Chọn bảng DynamoDB để làm việc
    table = dynamodb.Table('dev-db-buddycloud-detailresults')

    # Chuyển đổi hình ảnh thành Base64
    base64_image = convert_image_to_base64(image_path)

    # URL của API endpoint của FastAPI
    api_endpoint_url = 'http://192.85.4.149:8000/process_image_L2distance'

    # Gửi tệp hình ảnh lên API endpoint của FastAPI
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_endpoint_url, files=files)

    # Nhận kết quả từ API endpoint
    results_detail = response.json()

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
            "resultsDetail": results_detail,
            "rawImageURL": base64_image,
            "detectedImageURL": img1_base64  # Thay đổi giá trị tại đây
        }
    )

    ID_counter +=1

@app.post("/upload/")
async def create_upload_file(image: UploadFile = File(...), userName: str = Form(...), petName: str = Form(...), userPhoneNumber: str = Form(...), userEmail: str = Form(...), img1_base64: str = Form(...)):
    contents = await image.read()

    # Lưu tệp hình ảnh vào thư mục tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image_path = temp_image.name
        temp_image.write(contents)

    # Gọi hàm process_image_and_save_result để xử lý hình ảnh và lưu kết quả vào DynamoDB
    process_image_and_save_result(temp_image_path, userName, petName, userPhoneNumber, userEmail, img1_base64)

    # Xóa tệp hình ảnh tạm thời
    os.remove(temp_image_path)

    return {"message": "File uploaded successfully and results saved"}
