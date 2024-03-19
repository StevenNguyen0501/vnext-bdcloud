import boto3

# Kết nối với DynamoDB
dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                          aws_secret_access_key='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',
                          region_name='ap-southeast-1')

# Chọn bảng DynamoDB để làm việc
table = dynamodb.Table('dev-db-buddycloud-detailresults')

# Chuẩn bị dữ liệu để lưu
data_to_save = (
        {
            "ID": "1",
            "userID": "1",
            "userName": "Nguyen A",
            "petName": "Dog1",
            "petID": "1",
            "userPhoneNumber": "9993222",
            "userEmail": "example1@gmail.com",
            "timestamp": "15:48",
            "resultsID": "1",
            "resultsDetail": "A",
            "rawImageURL": "cbasdadads",
            "detectedImageURL": "1cbasdadads"
        },
        {
            "ID": "2",
            "userID": "2",
            "userName": "Nguyen Van B",
            "petName": "Dog2",
            "petID": "2",
            "userPhoneNumber": "9993223",
            "userEmail": "example2@gmail.com",
            "timestamp": "16:48",
            "resultsID": "2",
            "resultsDetail": "B",
            "rawImageURL": "cbasdadads1",
            "detectedImageURL": "2basdadads"
        },
        {
            "ID": "3",
            "userID": "3",
            "userName": "Nguyen C",
            "petName": "Dog3",
            "petID": "3",
            "userPhoneNumber": "9993224",
            "userEmail": "example3@gmail.com",
            "timestamp": "17:48",
            "resultsID": "3",
            "resultsDetail": "C",
            "rawImageURL": "cbasdadads2",
            "detectedImageURL": "3basdadads"
        },
        {
            "ID": "4",
            "userID": "4",
            "userName": "Pham Van Q",
            "petName": "Dog4",
            "petID": "4",
            "userPhoneNumber": "9993225",
            "userEmail": "example4@gmail.com",
            "timestamp": "18:48",
            "resultsID": "4",
            "resultsDetail": "D",
            "rawImageURL": "cbasdadads3",
            "detectedImageURL": "4basdadads"
        },
        {
            "ID": "5",
            "userID": "5",
            "userName": "Pham Van K",
            "petName": "Dog5",
            "petID": "5",
            "userPhoneNumber": "9993226",
            "userEmail": "example5@gmail.com",
            "timestamp": "19:48",
            "resultsID": "5",
            "resultsDetail": "A",
            "rawImageURL": "cbasdadadse",
            "detectedImageURL": "5basdadadse"
        },
        {
            "ID": "6",
            "userID": "6",
            "userName": "Le Dinh N",
            "petName": "Dog6",
            "petID": "6",
            "userPhoneNumber": "9993227",
            "userEmail": "example6@gmail.com",
            "timestamp": "20:48",
            "resultsID": "6",
            "resultsDetail": "B",
            "rawImageURL": "cbasdadadsd",
            "detectedImageURL": "6basdadadsd"
        },
        {
            "ID": "7",
            "userID": "7",
            "userName": "Vo Van",
            "petName": "Dog7",
            "petID": "7",
            "userPhoneNumber": "9993228",
            "userEmail": "example7@gmail.com",
            "timestamp": "21:48",
            "resultsID": "7",
            "resultsDetail": "C",
            "rawImageURL": "cbasdadadsc",
            "detectedImageURL": "7basdadadsc"
        },
        {
            "ID": "8",
            "userID": "8",
            "userName": "Nguyen Phuc",
            "petName": "Dog8",
            "petID": "8",
            "userPhoneNumber": "9993229",
            "userEmail": "example8@gmail.com",
            "timestamp": "22:48",
            "resultsID": "8",
            "resultsDetail": "D",
            "rawImageURL": "cbasdadadsee",
            "detectedImageURL": "8asdadadsee"
        },
        {
            "ID": "9",
            "userID": "9",
            "userName": "Le Giang",
            "petName": "Dog9",
            "petID": "9",
            "userPhoneNumber": "9993230",
            "userEmail": "example9@gmail.com",
            "timestamp": "23:48",
            "resultsID": "9",
            "resultsDetail": "A",
            "rawImageURL": "cbasdadadshh",
            "detectedImageURL": "9asdadadshh"
        },
        {
            "ID": "10",
            "userID": "10",
            "userName": "Le Mau",
            "petName": "Dog10",
            "petID": "10",
            "userPhoneNumber": "9993231",
            "userEmail": "example10@gmail.com",
            "timestamp": "0:48",
            "resultsID": "10",
            "resultsDetail": "B",
            "rawImageURL": "cbasdadadsj",
            "detectedImageURL": "10basdadadsj"
        },
        {
            "ID": "11",
            "userID": "11",
            "userName": "Le Hong",
            "petName": "Dog11",
            "petID": "11",
            "userPhoneNumber": "9993232",
            "userEmail": "example11@gmail.com",
            "timestamp": "1:48",
            "resultsID": "11",
            "resultsDetail": "C",
            "rawImageURL": "cbasdadadsk",
            "detectedImageURL": "11basdadadsk"
        },
        {
            "ID": "12",
            "userID": "12",
            "userName": "Nguyen Viet C",
            "petName": "Dog12",
            "petID": "12",
            "userPhoneNumber": "9993233",
            "userEmail": "example12@gmail.com",
            "timestamp": "2:48",
            "resultsID": "12",
            "resultsDetail": "D",
            "rawImageURL": "cbasdadadskkk",
            "detectedImageURL": "12sdadadskkk"
        },
        {
            "ID": "13",
            "userID": "13",
            "userName": "Hoang Thi F",
            "petName": "Dog13",
            "petID": "13",
            "userPhoneNumber": "9993234",
            "userEmail": "example13@gmail.com",
            "timestamp": "3:48",
            "resultsID": "13",
            "resultsDetail": "A",
            "rawImageURL": "cbasdadadsl",
            "detectedImageURL": "13basdadadsl"
        },
        {
            "ID": "14",
            "userID": "14",
            "userName": "Hoang Dai",
            "petName": "Dog14",
            "petID": "14",
            "userPhoneNumber": "9993235",
            "userEmail": "example14@gmail.com",
            "timestamp": "4:48",
            "resultsID": "14",
            "resultsDetail": "B",
            "rawImageURL": "cbasdadads999",
            "detectedImageURL": "14sdadads"
        }
)
# Lưu từng mục dữ liệu vào DynamoDB
for item in data_to_save:
    response = table.put_item(Item=item)

# Kiểm tra và xác nhận
print("Dữ liệu đã được lưu vào DynamoDB.")
