import os

from conf.local import config  # local config
from conf.base import gcp_config  # cloud config

from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account

from pathlib import Path
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# the bigquery service class is for big query related services.


class BigQueryService:
    def __init__(self, service_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
        self.service_key_path = service_key_path
        self.client = bigquery.Client()

    def query_to_table(self, query, table_id):  # query to big query table
        job_config = bigquery.QueryJobConfig(destination=table_id)
        query_job = self.client.query(query, job_config)
        query_job.result()
        print("Query results loaded to the table {}".format(table_id))

    def query_to_df(self, query):  # query to pandas dataset local
        df = self.client.query(query).to_dataframe()
        return(df)

    def table_to_df(self, project_name, bq_dataset_name, bq_table_name):
        dataset_ref = self.client.dataset(
            bq_dataset_name, project=project_name)
        table_ref = dataset_ref.table(bq_table_name)
        table = self.client.get_table(table_ref)
        df = self.client.list_rows(table).to_dataframe()
        return(df)

    def query_gsheet_bq_to_df(self, project_name, query):
        scopes = ['https://www.googleapis.com/auth/drive',
                  'https://www.googleapis.com/auth/bigquery']
        credentials = service_account.Credentials.from_service_account_file(
            self.service_key_path, scopes=scopes)
        client = bigquery.Client(credentials=credentials, project=project_name)
        df = client.query(query).to_dataframe()
        return(df)

    def gcs_json_to_table(self, bq_dataset_name, gcs_file_uri, bq_table_prefix,
                          job_schema=None):
        dataset_ref = self.client.dataset(bq_dataset_name)
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        uri = gcs_file_uri
        file_name_without_extension = Path(uri).stem
        bq_table_name = bq_table_prefix + file_name_without_extension
        # check if the table exist
        if (bq_table_name in self.get_table_list(bq_dataset_name)):
            print("Replacing existing table {}".format(bq_table_name))
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        else:
            job_config.schema = job_schema

        load_job = self.client.load_table_from_uri(
            uri,
            dataset_ref.table(bq_table_name),
            location=config.gcs_location,
            job_config=job_config,
        )
        print("Starting job {}".format(load_job.job_id))
        load_job.result()
        print("Job finished.")
        destination_table = self.client.get_table(
            dataset_ref.table(bq_table_name))
        print("Loaded {} rows.".format(destination_table.num_rows))

    def get_table_list(self, bq_dataset_name):  # get table lists from a bq dataset
        dataset = self.client.get_dataset(bq_dataset_name)
        tables = list(self.client.list_tables(dataset))
        if tables:
            table_list = []
            for table in tables:
                table_list.append(table.table_id)
        return(table_list)


# This class is for google cloud storage related services
class GCSService:
    def __init__(self, service_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
        self.client = storage.Client()

    def upload_blob(self, bucket_name, sub_directory, destination_file_name, source_file_name):  # upload blob
        bucket = self.client.bucket(bucket_name)
        destination_blob_name = os.path.join(
            sub_directory, destination_file_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

    def download_blob(self, bucket_name, destination_file_name, source_file_name):  # download blob
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_file_name)
        blob.download_to_filename(source_file_name)

        print(
            "File {} downloaded from {}.".format(
                source_file_name, destination_file_name
            )
        )


# This class is for google drive related services
class GDriveService:
    def __init__(self, gdrive_key_path, scopes):
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            gdrive_key_path, scopes)
        self.client = gspread.authorize(creds)

    def sheet_to_df(self, gsheet_name):
        sheet = self.client.open(gsheet_name).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')\
            .str.replace('(', '_').str.replace(')', '_').str.replace(':', '').\
            str.replace('/', '_').str.replace('.', '').str.replace('#', 'num').\
            str.replace(',', '')
        return(df)
