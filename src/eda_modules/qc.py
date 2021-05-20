import os
import pandas as pd
import pandas_profiling
from .utility import v_print


# Currently the QC_core only generates pandas_profiling reports and save as files
# TODO, save the pandas_profiling objects, and return a list of these objects

# From these objects, create list of dataframes, such that, each dataframe, contains the number missingness, n_zeros ...

def qc_core(df_list: list, name_list: list, outputdir: str):
    """ QC

    Args:
        df_list [(pd.DataFrame)]: list with merged dataframe from feature tables and one target variable table
        name_list: list of names to call output (one for each df df_list)
        outputdir (str): where EDA reports are saved
    """
    assert len(df_list) == len(name_list)

    for df, table_name in zip(df_list, name_list):
        vars = {'cat': {'check_composition': False}}
        profile_full = pandas_profiling.ProfileReport(
            df, title=table_name + "exploration", vars=vars, minimal=len(df.columns) > 7)
        reportfilepath = os.path.join(outputdir, table_name + '_full.html')
        profile_full.to_file(output_file=reportfilepath)


def qc_zeros(df_list: list, col_list: list):
    for df in df_list:
        for x in col_list:
            if x in df.columns:
                print(x + "\t" + str(sum(df[x] == 0)))


def qc_to_csv(df, output_dir, filename):
    if not os.path.exists(os.path.join(output_dir, "qc")):
        os.mkdir(os.path.join(output_dir, "qc"))

    vars = {'cat': {'check_composition': False}}
    profile_full = pandas_profiling.ProfileReport(
        df, title="exploration", vars=vars, minimal=len(df.columns) > 7)

    dict_list = []
    idx_list = []
    for x in df.columns:
        if df[x].dtypes == 'float64':
            idx_list.append(x)
            dict_list.append({key: profile_full.description_set["variables"][x][key]
                              for key in ['n', 'is_unique', 'distinct_count_with_nan', 'distinct_count_without_nan', 'distinct_count',
                                          'n_unique', 'p_unique',
                                          'n_missing', 'p_missing',
                                          'n_infinite', 'p_infinite',
                                          'n_zeros', "p_zeros"]})
    ret = pd.DataFrame(dict_list, index=idx_list)
    ret.to_csv(os.path.join(output_dir, "qc", filename))
