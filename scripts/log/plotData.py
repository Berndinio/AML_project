import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
lossList = [0.011966892518103123, 0.009469014592468739, 0.009421924129128456, 0.009644722566008568, 0.00993915181607008, 0.009010311216115952, 0.009399214759469032, 0.008346717804670334, 0.008661427535116673, 0.009176795370876789]
recList = [0, 0.25, 0.08333333333333333, 0.08333333333333333, 0, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
accList = [0.9769230769230769, 0.9807692307692307, 0.9788461538461538, 0.9769230769230769, 0.9769230769230769, 0.9788461538461538, 0.9788461538461538, 0.9807692307692307, 0.9788461538461538, 0.9788461538461538]
preList = [0, 0.75, 1.0, 0.5, 0, 1.0, 1.0, 1.0, 0.6666666666666666, 0.6666666666666666]
epochList = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]



def plotter(x, y, title):
    #x_smooth = np.linspace(x[0],x[-1],50)
    #y_smooth = spline(x,y,x_smooth)
    x_smooth = x
    y_smooth = y
    plt.plot(x_smooth, y_smooth)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title+"-Epoch plot")
    plt.savefig("data/plots/"+title+"_vanillaNN.png")
    plt.clf()

plotter(epochList, lossList, "Loss")
plotter(epochList, recList, "Recall")
plotter(epochList, accList, "Accuracy")
plotter(epochList, preList, "Precision")
