import boto3
import json

dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id='AKIA47CRWLV57NUYSDTM',
                          aws_secret_access_key='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',
                          region_name='ap-southeast-1')

# Chọn bảng DynamoDB để làm việc
table = dynamodb.Table('dev-db-buddycloud-detailresults')

response = table.scan()

data = response['Items']

while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    data.extend(response['Items'])

# Convert data thành JSON
json_data = json.dumps(data)
print(json_data)
