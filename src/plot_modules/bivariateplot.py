from seaborn.palettes import color_palette
from sklearn.utils import validation
from .baseplot import baseplot
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


class bivariateplot(baseplot):
  def __init__(self, df:pd.DataFrame,
                     x_variable:str = None,
                     y_variable:str =  None,
                     color_variable:str = None,
                     xlab:str = None,
                     ylab:str = None,
                     title:str = None,
                     addition_txt:str = "",
                     fig:matplotlib.figure.Figure = None,
                     axes:matplotlib.axes.Axes = None):
    super().__init__(fig, axes)
    self.x_variable = x_variable
    self.y_variable = y_variable
    self.color_variable = color_variable
    self.data = df
    self.x = self.data[self.x_variable]
    self.y = self.data[self.y_variable]
    self.addition_txt = addition_txt
    self.title = title
    self.xlable = x_variable if xlab is None else xlab
    self.ylable = y_variable if ylab is None else ylab
    if self.color_variable is not None:
      if "p_change_c" in self.color_variable:
        self.color_dict_in_use = self.process_color_dict
      elif "agitation" in self.color_variable:  
        self.color_dict_in_use = self.aggitation_color_dict
      else:
        if self.data[self.color_variable].dtype.name == 'category':
          keys = self.data[self.color_variable].unique().tolist()
          for cat in self.data[self.color_variable].cat.categories:
            # Categories may have other values that are not present in the data
            if cat not in keys:
              keys.append(cat)
          selected_palette = sns.color_palette("bright", n_colors=len(keys))
        else:
          norm = matplotlib.colors.Normalize(vmin=np.nanmin(self.data[self.color_variable]), 
          vmax=np.nanmax(self.data[self.color_variable]), clip=True)
          mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)
          keys = self.data[self.color_variable].unique()
          selected_palette = [mapper.to_rgba(val) for val in keys]
        
        self.color_dict_in_use = dict(zip(keys, selected_palette))
    else:
      self.color_dict_in_use = None

  
  def boxswarmplot(self):
    print(self.x)
    self.ax = sns.boxplot(x = self.x, y=self.y)
    self.ax.set_xticklabels(self.ax.get_xticks(), rotation = 90)

    self.fig = sns.swarmplot(x=self.x, y=self.y, ax = self.ax, alpha=0.7, edgecolor="gray", linewidth=1).get_figure()

    self.fig.subplots_adjust(bottom=0.25)
    return self

  def swarmplot(self):
    self.fig = sns.swarmplot(x=self.x, y=self.y).get_figure()
    return self

  def set_color(self):
    if self.color_variable is None:
      print("color_variable is unset!")
      self.color = None
    else:
      self.color = [self.color_dict_in_use[key] for key in self.data[self.color_variable]]
    return self

  def pi_ts_plt(self):
    self.fig = plt.figure(figsize=(200,100)) 
    plt.rcParams['date.epoch'] = '0000-12-31'
    
    self.axes.plot_date(self.x, self.y, "-", color = "grey", linewidth=1, alpha=.7)
    self.axes.xaxis.set_major_locator(mdates.YearLocator())
    self.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    mean_y = np.mean(self.y)
    sd_y = np.std(self.y)
    self.axes.axhline(y=mean_y+3.89*sd_y, color = "grey", linestyle = "dotted")
    self.axes.axhline(y=mean_y-3.89*sd_y, color = "grey", linestyle = "dotted")
    self.axes.axhline(y=mean_y, color = "grey", linestyle = "dotted")
    self.axes.update(dict(ylabel=self.ylable))

    return self

  def pi_top_vs_bottom_ts(self, color="blue", label=""):
    self.axes.plot(self.x, self.y, color = color, linewidth=1, alpha=.7, label=label)
    self.axes.legend()


  def tsplot(self, add_legend=False):
    self.fig = plt.figure(figsize=(200,100)) 
    plt.rcParams['date.epoch'] = '0000-12-31'
    self.axes.plot(self.x, self.y, color = "grey", linestyle = "dotted", linewidth=1, alpha=.7)
    self.axes.set_xlim(min(self.x),max(self.x))
    self.axes.scatter(self.x, self.y, s = 8, c = self.color, cmap="Set3", alpha=.7)
    mean_y = np.mean(self.y)
    sd_y = np.std(self.y)
    self.axes.axhline(y=mean_y+3.89*sd_y, color = "grey", linestyle = "dotted")
    self.axes.axhline(y=mean_y-3.89*sd_y, color = "grey", linestyle = "dotted")
    self.axes.axhline(y=mean_y, color = "grey", linestyle = "dotted")
    self.axes.update(dict(ylabel=self.ylable))
    
    if add_legend and self.color_dict_in_use is not None:
      # The following two lines generate custom fake lines that will be used as legend entries:
      markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in self.color_dict_in_use.values()]
      self.axes.legend(markers, self.color_dict_in_use.keys(), numpoints=1, bbox_to_anchor=(1.01, 1), loc='upper left')
      box = self.axes.get_position()
      self.axes.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    
    return self

  def axvline(self,**kwargs):
    self.axes.axvline(**kwargs)
    return self

  def lmplot(self):
    plt.close(self.fig)
    g = sns.lmplot(x=self.x_variable, y=self.y_variable, data=self.data, hue = self.color_variable, palette=self.color_dict_in_use)
    self.fig = g.fig
    return self

  def scatterhistplot(self):
    plt.close(self.fig)
    y_range = max(self.y) - min(self.y)

    if self.x.dtype.name != 'category':
      x_range = max(self.x) - min(self.x)
      g = sns.JointGrid(x=self.x, y=self.y, data=self.data,
        xlim = (min(self.x) - .1*x_range, max(self.x) + .2*x_range),
        ylim = (min(self.y) - .1*y_range, max(self.y) + .2*y_range))
      
      g = g.plot_joint(sns.regplot, scatter_kws={'facecolors':"#FFFFFF"}, line_kws={'color':'purple'})
      g.ax_joint.text(min(self.x), max(self.y), self.addition_txt, fontstyle='italic')

    else:
      x_range = len(self.x.cat.categories)
      g = sns.JointGrid(x=self.x, y=self.y, data=self.data, xlim = (- .1*x_range,1.2*x_range), ylim = (min(self.y) - .1*y_range, max(self.y) + .2*y_range))
      for tick in g.ax_joint.get_xticklabels():         
        tick.set_rotation(90)
      g.fig.subplots_adjust(bottom=0.2)


    if self.color_dict_in_use is not None:
      selected_palette = self.color_dict_in_use
      g = g.plot_joint(sns.scatterplot, hue = self.color_variable, style = self.color_variable, data = self.data, markers=self.filled_markers, alpha=.5, palette=selected_palette)
      for key in self.color_dict_in_use:
        sns.kdeplot(self.data.loc[self.data[self.color_variable]==key, self.x_variable], bw = x_range/20, ax=g.ax_marg_x, legend=False, color = self.color_dict_in_use[key])
        sns.kdeplot(self.data.loc[self.data[self.color_variable]==key, self.y_variable], bw = y_range/20, ax=g.ax_marg_y, legend=False, vertical=True, color = self.color_dict_in_use[key])
      plt.legend(frameon=True)
    
    else:
      selected_palette = sns.color_palette("Spectral")
      g = g.plot_joint(sns.scatterplot, hue = self.color_variable, data = self.data, markers=self.filled_markers, alpha=.5, palette=selected_palette, legend=False)
    
    self.fig = g.fig
    return self

  def line_plot(self,**kwargs):
    self.axes.plot(self.x, self.y,**kwargs)
    return self

  def scatter_plot(self,**kwargs):
    if self.color_variable is not None:
      self.set_color()
      self.axes.scatter(self.x, self.y, color=self.color, **kwargs)
    else:
      self.axes.scatter(self.x, self.y, **kwargs)
    self.set_x_label(self.xlable).set_y_label(self.ylable)
    return self