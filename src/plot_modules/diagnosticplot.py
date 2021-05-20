from .baseplot import baseplot
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
from itertools import compress

class diagnosticplot(baseplot):
  def __init__(self, df:pd.DataFrame,
                     y_col:str = None,
                     y_hat_col:str =  None,
                     color_variable:str = None,
                     title:str = None,
                     addition_txt:str = "",
                     fig:matplotlib.figure.Figure = None,
                     axes:matplotlib.axes.Axes = None):
    super().__init__(fig, axes)
    self.y_col = y_col
    self.y_hat_col = y_hat_col
    self.color_variable = color_variable
    self.data = df
    self.y = self.data[self.y_col]
    self.y_hat = self.data[self.y_hat_col]
    self.residual = self.y - self.y_hat
    self.addition_txt = addition_txt
    self.title = title
    if self.color_variable is not None:
      self.color_dict_in_use = self.year_color_dict if "year" in self.color_variable else self.wcb_color_dict
    if self.color_variable == "wcb_sparge":
      self.color_dict_in_use = self.wcb_sparge_color_dict


  def set_color(self):
    if self.color_variable is None:
      raise Exception("color_variable is unset!")
    self.color = [self.color_dict_in_use[key] for key in self.data[self.color_variable]]
    # self.legend =
    return self


  def get_normalized_residual(self):
    def normalize_list(x):
        m = np.mean(x)
        sd = np.std(x)
        return [(i-m)/sd for i in x]

    residual_list = self.residual.to_list()
    self.data["normalized_residual"] = normalize_list(residual_list)


  def qqnorm(self, vs_fitted = False, num_of_run_to_label = 10):
    def get_theoretical_q(sample_size):
        p_array = np.linspace(1/(1+sample_size), 1, sample_size, endpoint=False)
        return [scipy.stats.norm.ppf(x) for x in p_array]

    self.get_normalized_residual()
    theoretical_q = get_theoretical_q(len(self.data.normalized_residual))

    self.data = self.data.sort_values(by = ["normalized_residual"], ascending=True)

    x = theoretical_q
    if vs_fitted:
        x = self.data[self.y_hat_col].to_list()

    self.set_color()
    for key in self.color_dict_in_use:
        if key == "None":
            continue
        idx = [ cc == self.color_dict_in_use[key] for cc in self.color]
        self.axes.scatter(list(compress(x, idx)), list(compress(self.data.normalized_residual, idx)), c = list(compress(self.color, idx)), label=key)
    self.axes.legend(loc = "lower right")
    self.axes.update(dict(ylabel="Normalized residual"))
    self.axes.update(dict(title = self.title))

    if vs_fitted:
        self.axes.update(dict(xlabel="Fitted values"))
    else:
        self.axes.update(dict(xlabel="Theoretical quantiles"))

    paired_points = list(zip(x, self.data.normalized_residual, self.data.run_number))
    leftright = 0
    for xx, yy, label in (paired_points[:num_of_run_to_label] + paired_points[-num_of_run_to_label:]):
        leftright = leftright + 1
        self.axes.text(xx, yy, label, fontsize=10, ha = "left" if leftright % 2 == 0 else "right")
    return self


