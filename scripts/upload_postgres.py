import os

import boto3
import click
from dotenv import load_dotenv


def upload_file_to_s3(file_path: str, bucket: str, key: str) -> None:
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )
    s3_client.upload_file(file_path, bucket, key)


@click.command()
@click.option(
    "--file-path", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to the file to upload."
)
@click.option("--key-name", type=str, required=True, help="S3 object key name.")
@click.option("--bucket-name", type=str, required=True, help="S3 bucket name.", default="autorag-research")
def main(file_path: str, key_name: str, bucket_name: str) -> None:
    load_dotenv()
    upload_file_to_s3(file_path, bucket_name, key_name)


if __name__ == "__main__":
    main()
