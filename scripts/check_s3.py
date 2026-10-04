import boto3
from dotenv import load_dotenv
 
load_dotenv()
boto3.client("s3").head_bucket(Bucket="trial-conversion-artifacts-adel")
print("credentials work")