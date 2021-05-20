from datetime import datetime

from pandas.core.frame import DataFrame
from src.services.gcp import BigQueryService
from conf.base import gcp_config
import numpy as np
from functools import reduce
from .utility import v_print

import pandas as pd
pd.options.mode.chained_assignment = None


def fetch_data(table_name: str, config_key: str, bq_dataset: str = "bsl_features"):
    """[summary]

        Args:
        table_name (str): [description]
        config_key (str): [description]

    Returns:
        [type]: [description]
    """
    query = "SELECT * FROM `pti-reindeer-sb-001-6e165697." + bq_dataset + "." + table_name + "`;"
    print("Use BQ")
    print(query)
    service=BigQueryService(config_key)
    return service.query_to_df(query)


def merge_target_table(df, target_column_list, target_table_name, target_bq_dataset,credential, merge_key = "run_number"):
    """[summary]
        Args:
        df (pandas dataframe)
        target_table_name (str)
        target_bq_dataset (str)
        credential (str): 

    Returns:
        [dataframe]: [description]
    """
    target_table = fetch_data(target_table_name, credential, target_bq_dataset)
    df_with_target = target_table.loc[:,target_column_list].merge(df, on =merge_key, how="left",indicator=False)
    df_with_target.reset_index(inplace=True)
    df_with_target = df_with_target.drop(columns=['index'])
    return(df_with_target)


def merge_run_info_table(df, run_info_table_name, run_info_bq_dataset,credential, merge_key="run_number"):
    """[summary]
        Args:
        df (pandas dataframe)
        run_info_table_name (str)
        run_info_bq_dataset (str)
        credential (str): 

    Returns:
        [dataframe]: [description]
    """
    run_info_table = fetch_data(run_info_table_name, credential, run_info_bq_dataset)
    df_with_run_info = df.merge(run_info_table, on =merge_key, how="left",indicator=False)
    df_with_run_info.reset_index(inplace=True)
    df_with_run_info = df_with_run_info.drop(columns=['index'])
    return(df_with_run_info)


def remove_run_1_30(df):
    """[summary]
        Args:
        df (pandas dataframe)

    Returns:
        [dataframe]: [description]
    """ 
    for i in range(1,31,1): 
        run = "RUN"+str(i)
        df = df[df.run_number != run]
    return (df)
 

def add_flag_for_run325_and_after(df):
    """[summary]
        Args:
        df (pandas dataframe)

    Returns:
        [dataframe]: [description]
    """
    df.sort_values(by=['sort_order'], inplace=True)
    #find the order index for run325
    df.loc[df['sort_order']>=364, 'run_325_and_after'] = 1
    df.run_325_and_after[~(df['sort_order']>=364)] = 0
    return (df) 

def run_all_actions(feature_table_name, feature_table_data_set, credential):
    df = fetch_data(feature_table_name, credential, feature_table_data_set)
    df = merge_target_table(df,gcp_config.target_column_list,gcp_config.target_table_name,gcp_config.bq_primary,credential)
    df = merge_run_info_table(df, gcp_config.run_info_table_name, gcp_config.bq_primary,credential)
    df = remove_run_1_30(df)
    df = add_flag_for_run325_and_after(df)
    return (df)

    
