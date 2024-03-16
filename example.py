from fastapi import FastAPI, HTTPException
import boto3

# Khởi tạo FastAPI
app = FastAPI()
# Kết nối với DynamoDB
def connect_to_dynamodb():

    dynamodb = boto3.resource('dynamodb',
                              aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                              aws_secret_access_key='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',
                              region_name='ap-southeast-1')

    # Chọn bảng DynamoDB để làm việc
    table = dynamodb.Table('dev-db-buddycloud-detailresults')
    return dynamodb, table

@app.get("/item/{id}/{timestamp}")
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
        item = response.get('Item')

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
@app.delete("/items/{item_id}")
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