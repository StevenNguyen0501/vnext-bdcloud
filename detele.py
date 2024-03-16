from fastapi import FastAPI, HTTPException
from boto3.dynamodb.conditions import Key
import boto3

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Kết nối với DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id='AKIA47CRWLV57NUYSDTM',
    aws_secret_access_key='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',
    region_name='ap-southeast-1'
)

# Chọn bảng DynamoDB để làm việc
table = dynamodb.Table('dev-db-buddycloud-detailresults')

# Định nghĩa endpoint để xóa một mục từ DynamoDB
@app.delete("/items/{item_id}")
async def delete_item(item_id: str, timestamp: str):
    try:
        response = table.delete_item(
            Key={
                'ID': item_id,
                'timestamp': timestamp
            }
        )
        return {"message": "Item deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
