import matplotlib.pyplot as plt
import matplotlib


class baseplot(object):
  def __init__(self, fig:matplotlib.figure.Figure = None,
                     axes:matplotlib.axes.Axes = None):
    self.process_color_dict = dict({
                      "start":'#000000',
                      "Drawdown optimization, pH slope checks TW-735585":'#1f77b4',
                      "New spectrophotometer for OD measurement":'#ffc512',
                      "Centrifuge Discharge Calibration tightening (TW-1364152)":'#2ca02c',
                      "Drawdown optimization TW-1612538":'#d62728',
                      "Agitator high vibration hence run at lower speed":'#9467bd',
                      "Harvest at 75.5 hours for runs 353 onwards (routine is 72 hours)":'#8c564b',
                      "Media antifoam at lower limit 0.909kg": '#e377c2'
                      })
    self.aggitation_color_dict = dict({
                      "300":'#1f77b4',
                      "270":'#ff7f0e',
                      "250":'#2ca02c',
                      "290": '#d62728',
                      "325":'#9467bd',
                      "None": "#000000"})
    self.filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    if (fig == None) & (axes == None):
      self.fig, self.axes = plt.subplots(1)
    else:
      self.fig = fig
      self.axes = axes

  def set_x_label(self, xlab:str):
    self.axes.set_xlabel(xlab)
    return self

  def set_y_label(self, ylab:str):
    self.axes.set_ylabel(ylab)
    return self


  def savefig(self, figurename:str):
    self.fig.savefig(figurename)
    plt.close(self.fig)
