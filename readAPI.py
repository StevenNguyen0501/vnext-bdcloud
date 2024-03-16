import boto3


def scan_dynamodb_table(table_name):
    # Kết nối với DynamoDB
    dynamodb = boto3.resource('dynamodb',
                              aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                              aws_secret_access_key='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',
                              region_name='ap-southeast-1')
    # Chọn bảng DynamoDB để làm việc
    table = dynamodb.Table('dev-db-buddycloud-detailresults')

    response = table.scan()
    items = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    return items


table_name = 'dev-db-buddycloud-detailresults'  # Thay thế 'tên_bảng_của_bạn' bằng tên thực của bảng DynamoDB của bạn
all_items = scan_dynamodb_table(table_name)

# In ra toàn bộ dữ liệu
for item in all_items:
    print(item)
