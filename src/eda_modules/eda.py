from .utility import v_print, check_path, list_flatten

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib

from src.plot_modules.univariateplot import univariateplot
from src.plot_modules.bivariateplot import bivariateplot

from scipy import stats
import numpy as np
from collections.abc import Iterable


def r2(x, y):
  df = pd.DataFrame(zip(x,y))
  df = df.replace([np.inf, -np.inf], np.nan).dropna()
  ret = stats.pearsonr(df.iloc[:,0], df.iloc[:,1])[0] ** 2
  return ret


def auto_correlation(df_list:list, col_list:list, output_dir:str, dir_extra:str='auto_correlation'):
  """ Generate auto_correlation figures for each column
  Args:
      df_list (pd.DataFrame): Input dataframe
      output_dir (str): Output directory
      cols: list of the variables you want auto correlation for (assumes only 1 df in df_list)
  """
  outpath = check_path(output_dir, "auto_correlation", dir_extra)
  for df in df_list:
    for col in col_list:
      if col in list(df.columns):
        print(col)
        df = df.dropna(subset=[col])
        univariateplot(df, x_variable = col, title = col). \
          auto_correlation(). \
          savefig(os.path.join(outpath, col + '_auto_correlation.png'))
        univariateplot(df, x_variable = col, title = col). \
          partial_auto_correlation(). \
          savefig(os.path.join(outpath, col + '_partial_auto_correlation.png'))

def plot_obj_vs_time(df:pd.DataFrame, x:str, y:str, outpath:str):
  """ Create beeswarm plots for variable y at x time resolution
  Args:
      df (pd.DataFrame): Input dataframe
      x (str): Time resolution
      y (str): Variable of interest
      output_dir (str): Output directory
  """
  bivariateplot(df, x_variable = x, y_variable = y, title = y). \
    swarmplot(). \
    savefig(os.path.join(outpath, y + '_vs_' + x + '.png'))

def variable_distribution(df_list:list, col_list:list, output_dir:str, dir_extra:str):
  """ Create beeswarm plot for variables
  Args:
      df (pd.DataFrame): Input dataframe
      output_dir (str): Output directory
      cols:
  """
  outpath = check_path(output_dir, "distribution", dir_extra)
  for df in df_list:
    for col in col_list:
      if col in list(df.columns):
        print(col)
        univariateplot(df, x_variable = col). \
          boxswarmplot(). \
          savefig(os.path.join(outpath, col + '_swarmplot.png'))
        if "year" in list(df.columns):
          plot_obj_vs_time(df, x = "year", y = col, outpath = outpath)
        if "month" in list(df.columns):
          plot_obj_vs_time(df, x = "month", y = col, outpath = outpath)
        if "hour" in list(df.columns):
          plot_obj_vs_time(df, x = "hour", y = col, outpath = outpath)
  v_print("variable_distribution")


def scatter_plot(df_list:list, x_variables:list, y_variables:list, output_dir:str, dir_extra:str, threshold:float=.15, color_variable = "p_change_c"):
  """ Scatter plot with kernel estimation on x, y-axis

  Args:
      df (pd.DataFrame): Input dataframe
      output_dir (str): Output directory
  """
  outpath = check_path(output_dir, "scatter_plot", dir_extra)
  for df in df_list:
    for x in x_variables:
      for y in y_variables:
        # TODO need to work on y names, basically y is one of ccf measurement, or measures that are coming from the same stage ...

        if y not in list(df.columns):
          print ("y not in df")
          continue
        if (x != y) & (df[y].dtypes == 'float64') & (x in list(df.columns)):
          if color_variable is not None:
            plotting_df = df[[x, y, color_variable]]
          else:
            plotting_df = df[[x, y]]
          plotting_df = plotting_df.replace([np.inf, -np.inf], np.nan).dropna()
          if threshold is not None:
            current_r2 = r2(plotting_df[x], plotting_df[y])
            print(x, y, str(current_r2))

            pearson_df = pd.DataFrame(zip(plotting_df[x],plotting_df[y]))
            pearson_corr = stats.pearsonr(pearson_df.iloc[:,0], pearson_df.iloc[:,1])
            print(f"Pearson correlation: {pearson_corr}")

            if (current_r2 < threshold):
              continue

            addition_txt = 'r^2 = ' + str(current_r2)
          else:
            addition_txt = ""
          bivariateplot(plotting_df, x_variable = x, y_variable = y, color_variable = color_variable, addition_txt = addition_txt). \
            scatterhistplot(). \
            savefig(os.path.join(outpath, x + '_vs_' + y + '_scatter.png'))

def ts_plots(df_list:list, col_list:list, color_variable:str, output_dir:str, dir_extra:str='time_series'):
  """ Time series plots
  Args:
      df (pd.DataFrame): [description]
      output_dir (str): [description]
  """
  outpath = check_path(output_dir, "time_series", dir_extra)
  for df in df_list:
    for col in col_list:
      if (col in list(df.columns)) & ("year" in list(df.columns)):
        mean_y = np.mean(df[col])
        sd_y = np.std(df[col])
        print(df.loc[df[col] > mean_y+3.89*sd_y][["run_number", "date", col]])
        print(df.loc[df[col] < mean_y-3.89*sd_y][["run_number", "date", col]])

        plotting_df = df[["year", "date", col, "year_str"]]
        plotting_df = plotting_df.replace([np.inf, -np.inf], np.nan).dropna()
        mean_x = np.mean(plotting_df[col])
        sd_x = np.std(plotting_df[col])
        print(col)
        plot_ts_obj(plotting_df, col, color_variable, None).savefig(os.path.join(outpath, col + '_vs_time_colored_by_yr.png'))
    v_print("ts")


def ts_panel_figure(df:pd.DataFrame, list_of_col_list:list, color_variable:str, output_dir:str, dir_extra:str):
  df = df.replace([np.inf, -np.inf], np.nan).dropna()
  outpath = check_path(output_dir, "time_series", dir_extra)
  ct = 0
  for col_list in list_of_col_list:
    if len(col_list) == 0:
      continue
    ts_panel_figure_single(df, outpath, col_list, color_variable, dir_extra+'panel_'+str(ct))

    ct = ct+1


def ts_panel_figure_single(df:pd.DataFrame, output_dir:str, col_list:list, color_variable = "p_change_c", title:str='Final PCV'):
  if not os.path.exists(os.path.join(output_dir, "time_series")):
      os.mkdir(os.path.join(output_dir, "time_series"))
  plt.figure(figsize=(150.00, 150.00), dpi = 100)
  if not isinstance(color_variable, list) and not isinstance(color_variable, np.ndarray):
    color_variable = [color_variable]
  nrows = np.maximum(len(color_variable), len(col_list))

  fig, axs = plt.subplots(nrows)
  if not isinstance(axs, np.ndarray):
    axs = np.array([axs])
  col_list.sort(reverse = True)
  fig.suptitle(title)

  for i_row in range(nrows):
      color = color_variable[np.minimum(i_row,len(color_variable)-1)]
      col = col_list[np.minimum(i_row,len(col_list)-1)]
      plot_ts_obj(df, col, color, axs[i_row])
  for ax in axs.flat:
    ax.set(xlabel='year')
  for ax in axs.flat:
    ax.label_outer()
  fig.set_size_inches(10, 10)
  fig.savefig(os.path.join(output_dir, "time_series", title+'.png'))
  plt.close()


def plot_ts_obj(df:pd.DataFrame, y:str, color_variable, axs:matplotlib.axes.Axes, add_legend:bool = False):
  ts_plot = bivariateplot(df, x_variable = "production_start_time_utc", y_variable = y, color_variable = color_variable, axes = axs). \
    set_color().tsplot(add_legend=add_legend)
  if "initial_agitation_speed_rpm" in df:
    print("Plotting vertical lines for agitation")
    
    df["agit_diff"] = df["initial_agitation_speed_rpm"].replace("None", "270").astype(int).diff()
    for _, agit_change in df[df["agit_diff"]!=0].iterrows():
      timestamp = agit_change['production_start_time_utc']
      new_agit = agit_change["initial_agitation_speed_rpm"]
      if timestamp is not pd.NaT:
        ts_plot = ts_plot.\
          axvline(x=timestamp, color=ts_plot.color_dict_in_use[new_agit], label = f"agitation_{new_agit}", alpha=0.5)
  return ts_plot

def pi_ts_plot(df_list:list, col_list:list, color_variable:str, output_dir:str, dir_extra:str):
  outpath = check_path(output_dir, "pi_time_series", dir_extra)

  fig, axs = plt.subplots(len(df_list))
  if not isinstance(axs, Iterable):
    axs = [axs]

  title = dir_extra+'panel'
  fig.suptitle(title)

  for i_df, df in enumerate(df_list[:-1]):
    col = col_list[i_df][0]
    bivariateplot(df, x_variable = "pi_timestamp", y_variable = col, axes = axs[i_df]). \
      pi_ts_plt()
    axs[i_df].set(xlabel='Date and Time')
    axs[i_df].label_outer()
  
  plot_ts_obj(df_list[-1], col_list[-1][0], color_variable, axs[-1], add_legend=False)

  fig.set_size_inches(10, 10)
  fig.savefig(os.path.join(outpath, title+'.png'))
  plt.close(fig)


def pi_top_vs_bottom_plot(
  df_pi:pd.DataFrame, x_col:str, y_col:str, df_target:pd.DataFrame, rankby_col:str, visualoutputpath:str, job_id:str, fraction:float = 0.10):
  """Plot of top x% vs bottom x% pi time series

  Args:
      df_pi (pd.DataFrame): dataframe with pi data
      x_col (str): column of the pi data x axis
      y_col (str): column to plot time series of
      df_target (pd.DataFrame): dataframe with pi data
      rankby_col (str): runs will be ranked by this column
      visualoutputpath (str): output subfolder
      job_id (str): output suffix
      fraction (float, optional): fraction of top/bottom time series to show. Defaults to 0.10.
  """
  outpath = check_path( visualoutputpath, "pi_top_vs_bottom", job_id)
  title = job_id+'top_vs_bottom'

  df_target_sorted = df_target.sort_values(rankby_col)
  # Translate fraction into a number
  x_num = int(np.ceil(fraction * len(df_target_sorted)))
  top_runs = df_target_sorted.iloc[-x_num:]["run_number"]
  bottom_runs = df_target_sorted.iloc[:x_num]["run_number"]

  percent_int = int(np.round(fraction*100))

  fig, axs = plt.subplots()
  fig.suptitle(title)

  for i_bottom_run, bottom_run in enumerate(bottom_runs):
    bottom_run_pi_data = df_pi[df_pi["run_number"] == bottom_run].sort_values(x_col).reset_index(drop=True)
    legend = f"Bottom {percent_int}%" if i_bottom_run==0 else None
    bivariateplot(bottom_run_pi_data, x_variable = x_col, y_variable = y_col, axes = axs). \
      pi_top_vs_bottom_ts("blue", legend)
  
  for i_top_run, top_run in enumerate(top_runs):
    top_run_pi_data = df_pi[df_pi["run_number"] == top_run].sort_values(x_col).reset_index(drop=True)
    legend = f"Top {percent_int}%" if i_top_run==0 else None
    bivariateplot(top_run_pi_data, x_variable = x_col, y_variable = y_col, axes = axs). \
      pi_top_vs_bottom_ts("red", legend)

  axs.set(xlabel=x_col, ylabel=y_col)

  fig.set_size_inches(10, 10)
  fig.savefig(os.path.join(outpath, title+'.png'))


def boxswarmplot(df_list:list, x_variables:list, y_variables:list, output_dir:str, dir_extra:str, color_variable = "p_change_c"):
  """ Boxplot plot with swarmplot overlayed

  Args:
      df_list (list of pd.DataFrame): Input dataframes
      output_dir (str): Output directory
  """
  outpath = check_path(output_dir, "boxswarm_plot", dir_extra)
  for df in df_list:
    for x in x_variables:
      for y in y_variables:
        # TODO need to work on y names, basically y is one of ccf measurement, or measures that are coming from the same stage ...

        if y not in list(df.columns):
          continue
        if (x != y) & (df[y].dtypes == 'float64') & (x in list(df.columns)):
          if color_variable is not None:
            plotting_df = df[[x, y, color_variable]]
          else:
            plotting_df = df[[x, y]]
          plotting_df = plotting_df.replace([np.inf, -np.inf], np.nan).dropna()
        
          bivariateplot(plotting_df, x_variable = x, y_variable = y, color_variable = color_variable). \
            boxswarmplot(). \
            savefig(os.path.join(outpath, x + '_vs_' + y + '_boxswarm.png'))  


def eda_collection(df_list:list, col_list:list, y_variables:list, actionlist:list, visualoutputpath:str, job_id:str, threshold:float=.15, color_variable = "p_change_c"):
#df_list is a list of dataframe; if you have only one dataframe, you will need to put []
#around the df.
#col_list is a list of list of columns in each dataframe; if you have only one dataframe, you will need to put []
#around the column list
  if ("auto_correlation" in actionlist):
    auto_correlation(df_list, list_flatten(col_list), visualoutputpath, job_id)

  if ("distribution" in actionlist):
    variable_distribution(df_list, list_flatten(col_list), visualoutputpath, job_id)

  if ("time_series" in actionlist):
    ts_plots(df_list, list_flatten(col_list), color_variable, visualoutputpath, job_id)
    ts_panel_figure(df_list[0], col_list, color_variable, visualoutputpath, job_id)

  if ("pi_time_series" in actionlist):
    pi_ts_plot(df_list, col_list, color_variable, visualoutputpath, job_id)

  if ("scatter_plot" in actionlist): # TODO need improvement, wcb color is defined after merge with timestampes.
    scatter_plot(df_list, list_flatten(col_list), y_variables, visualoutputpath, job_id, threshold, color_variable)

  if ("boxswarmplot" in actionlist):
    boxswarmplot(df_list, list_flatten(col_list), y_variables, visualoutputpath, job_id, color_variable)

  if ("reg_plot" in actionlist): # TODO need improvement, wcb color is defined after merge with timestampes.
    reg_plot(df_list, list_flatten(col_list), y_variables, visualoutputpath, job_id, threshold, color_variable)
