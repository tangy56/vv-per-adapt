import os.path
import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argcomplete


def list_flatten(list2d: list):
    return sum(list2d, [])


def v_print(verb_args: str):
    """ Print Verbose message

    Args:
      verb_args (str): message to be printed when verbose mode is turned on.
    """
    logging.debug(verb_args)


def check_path(root_dir: str, current_dir: str, dir_extra: str):
    dir_list = [root_dir, current_dir, dir_extra]
    outpath = ""
    for tmp_dir in dir_list:
        outpath = tmp_dir if outpath == "" else os.path.join(outpath, tmp_dir)
        print(outpath)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
    return outpath


def get_parser():
    """ Create a new ArgumentParser object.

    Returns:
      ArgumentParser: Command line interferace. For more details, call
        python3 .\src\eda_scripts\pcv_qc_eda.py -h
    """
    parser = ArgumentParser(
        __doc__,
        description="",
        usage="""
    Example:
      (1) For Jira ticket https://jira.gene.com/jira/browse/RNDR-350 time series plots

        python .\src\eda_scripts\pcv_qc_eda.py -a time_series -j final_pcv_viability

      (2) Both QC and all EDAs of Jira ticket https://jira.gene.com/jira/browse/RNDR-350

        python .\src\eda_scripts\pcv_qc_eda.py -dev -j final_pcv_viability

      (3) QC and EDA on each BQ tables

        python .\src\eda_scripts\pcv_qc_eda.py
      """,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        help="Increase output verbosity (e.g., -vv is more than -v)",
    )
    parser.add_argument(
        "-a",
        dest="action",
        default=None,
        choices=["qc", "auto_correlation", "distribution",
                 "time_series", "scatter_plot", "reg_plot", "zeros"],
        type=str,
        help="Action of the QC and EDA",
    )
    parser.add_argument(
        "-j",
        dest="jira",
        default=".",
        choices=["final_pcv_viability",  # https://jira.gene.com/jira/browse/RNDR-350
                 "initial_pcv_viability",  # https://jira.gene.com/jira/browse/RNDR-385
                 "prod_pcv_viability",
                 "offline_measurements_viablecells",  # https://jira.gene.com/jira/browse/RNDR-387
                 "offline_measurements_pco2",        # https://jira.gene.com/jira/browse/RNDR-387
                 "offline_measurements_glukose",     # https://jira.gene.com/jira/browse/RNDR-387
                 "offline_measurements_laktat",      # https://jira.gene.com/jira/browse/RNDR-387
                 "offline_measurements_osmolalit",   # https://jira.gene.com/jira/browse/RNDR-387
                 "cultivation_perf_pcv",
                 "cultivation_perf_viability",
                 "cultivation_perf_pco2",
                 "cultivation_perf_ph",
                 "cultivation_perf_laktat",
                 "cultivation_perf_glukose",
                 "cultivation_perf_viablecells",
                 "log_100minus_viability",
                 "delta_within_over_duration",
                 "log_pcv_ratio_over_duration",
                 "100minus_viability",
                 "delta_within",
                 "delta_between",
                 "thaw_durations",
                 "target_v"
                 ],
        type=str,
        help="Job associated with jira tickets",
    )
    parser.add_argument(
        "-dev",
        dest="dev",
        action="count",
        default=False,
        help="Enable development mode",
    )
    argcomplete.autocomplete(parser)
    return parser


def generate_model_input_and_rename_file(dest_dir,
                                         service_key: str,
                                         remove_extreme: bool = False,
                                         exclude_datetime: bool = False,
                                         encode_categorical: bool = False,
                                         scale_type: str = "simple",
                                         keep_run_number: bool = False,
                                         hypothesis_list: list = [],
                                         new_file_name: str = "new_filename.csv",
                                         cwd=os.getcwd()):
    hypothesis_flag = ""
    outfile_name = "final_model_input_all.csv"
    if len(hypothesis_list) > 0:
        hypothesis_flag = "--hypothesis_list='"+",".join(hypothesis_list)+"'"
        outfile_name = "final_model_input_" + \
            "_".join(hypothesis_list) + ".csv"

    cmd = "python3 " + os.path.join(cwd, "src", "eda_modules", "main_create_final_model_input.py") + " " + \
        "--credentials_path=" + service_key + " " + \
        "--local_destination=" + dest_dir + " " + \
        "--remove_extreme={} ".format(remove_extreme) + \
        "--exclude_datetime={} ".format(exclude_datetime) + \
        "--encode_categorical={} ".format(encode_categorical) + \
        "--scale_type={} ".format(scale_type) + \
        "--keep_run_number={} ".format(keep_run_number) + \
        hypothesis_flag
    print(cmd)
    os.system(cmd)
