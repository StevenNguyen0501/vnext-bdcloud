from fastapi import FastAPI, HTTPException
import boto3

# Khởi tạo FastAPI
app = FastAPI()
# Kết nối với DynamoDB
dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                          aws_secret_access_key='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',
                          region_name='ap-southeast-1')

# Chọn bảng DynamoDB để làm việc
table = dynamodb.Table('dev-db-buddycloud-detailresults')


@app.get("/item/{id}/{timestamp}")
async def read_item(id: str, timestamp: str):
    try:
        # Truy vấn dữ liệu từ bảng
        response = table.get_item(
            Key={
                'ID': id,
                'timestamp': timestamp
            }
        )
        item = response.get('Item')

        # Kiểm tra xem item có tồn tại hay không
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
