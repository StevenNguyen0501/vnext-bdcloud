import base64
from datetime import datetime
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import requests
import boto3
app = FastAPI()
def process_image_and_save_result(image_path: str, userName: str, petName: str, userPhoneNumber: str, userEmail: str, img1_base64: str, results_detail:dict):
    # Kết nối với DynamoDB
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
            "resultsDetail": results_detail,
            "rawImageURL": img1_base64,
            "detectedImageURL": image_path  # Thay đổi giá trị tại đây
        }
    )

    ID_counter +=1