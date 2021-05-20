from statsmodels.graphics import tsaplots
from .baseplot import baseplot
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


class univariateplot(baseplot):
  def __init__(self, df:pd.DataFrame,
                     x_variable:str = None,
                     xlab:str = None,
                     title:str = None,
                     fig:matplotlib.figure.Figure = None,
                     axes:matplotlib.axes.Axes = None):
    super().__init__(fig, axes)
    self.x = df[x_variable]
    self.title = title
    self.xlable = x_variable if xlab is None else xlab

  def auto_correlation(self):
    self.fig = tsaplots.plot_acf(self.x, title = self.title)
    return self

  def partial_auto_correlation(self):
      self.fig = tsaplots.plot_pacf(self.x, title = self.title)
      return self  

  def swarmplot(self):
    self.fig = sns.swarmplot(x=self.x).get_figure()
    return self

  def boxswarmplot(self):
    self.ax = sns.boxplot(x = self.x)
    self.fig = sns.swarmplot(x=self.x, ax = self.ax, color=".2").get_figure()
    return self