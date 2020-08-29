#from __future__ import division, print_function
import matplotlib
#print(matplotlib.get_cachedir())
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import scipy.stats
# In a notebook environment, display the plots inline
#%matplotlib inline


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    b = np.array([m, m-h, m+h])
    return b.clip(0)

# Set some parameters to apply to all plots. These can be overridden
# in each plot if desired

# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 30)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

# Define a function for the line plot with intervals
def lineplotCI(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
    # Shade the confidence interval
    ax.fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')

    plt.show()

def plot_all(data, ax, color_code):
    # Get the confidence intervals of the model
    data_stats = [mean_confidence_interval(data[:,i]) for i in range(data.shape[1])]
    data_stats = np.array(data_stats).T
    predict_mean_ci_low = data_stats[1,:]#np.min(data,axis=0)
    predict_mean_ci_upp = data_stats[2,:]#np.max(data,axis=0)
    # Data for regions where we want to shade to indicate the intervals has
    # to be sorted by the x axis to display correctly
    CI_df = pd.DataFrame(columns = ['x_data', 'low_CI', 'upper_CI'])
    CI_df['x_data'] = np.arange(start=0,stop=5,step=1)
    CI_df['low_CI'] = predict_mean_ci_low
    CI_df['upper_CI'] = predict_mean_ci_upp
    CI_df.sort_values('x_data', inplace = True)
    # Call the function to create plot
    #lineplotCI(x_data = np.arange(start=1,stop=101,step=1), 
    #           y_data = data_stats[0,:], 
    #           sorted_x = CI_df['x_data'], 
    #           low_CI = CI_df['low_CI'], 
    #           upper_CI = CI_df['upper_CI'], 
    #           x_label = 'Epoch',
    #           y_label = 'Train Loss',
    #           title = 'Train Loss for Elman-Tanh')

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(np.arange(start=0,stop=5,step=1), data_stats[0,:],
            lw = 1, color = color_code, alpha = 1)#, label = 'Mean')
    # Shade the confidence interval
    ax.fill_between(CI_df['x_data'], CI_df['low_CI'], CI_df['upper_CI'],
                    color = color_code, alpha = 0.4)#, label = '95% CI')
    # Label the axes and provide a title
    return ax
    


tda_max = np.array([[0.99889,0.41167,0.42500,0.41167,0.40833],
	                [0.99833,0.47167,0.39167,0.33667,0.36833],
	                [0.95333,0.41167,0.35000,0.33667,0.39000],
                    [1.00000,0.42833,0.36167,0.36333,0.40167],
                    [1.00000,0.43667,0.34833,0.34333,0.40500],
                    [0.99000,0.40667,0.36500,0.34167,0.41833],
                    [1.00000,0.39333,0.33500,0.35000,0.41667],
                    [0.98333,0.44833,0.35667,0.35167,0.43667],
                    [1.00000,0.43333,0.35167,0.34500,0.43500],
                    [0.96000,0.45000,0.36500,0.37167,0.44167],
                    [0.99333,0.46333,0.37333,0.37333,0.45667],
                    [0.97667,0.43333,0.39500,0.39500,0.49667],
                    [0.97667,0.50167,0.40500,0.41000,0.47833],
                    [0.98000,0.48000,0.37333,0.37833,0.44167],
                    [0.95667,0.53833,0.41333,0.38667,0.45333]])

tda_max_color = '#be0119'

tda_min = np.array([[0.99889,0.85333,0.79000,0.46167,0.36333],
	                [0.99833,0.86333,0.67167,0.52333,0.37000],
	                [0.95333,0.85333,0.69000,0.55500,0.38833],
                    [1.00000,0.85167,0.69000,0.56000,0.43333],
                    [1.00000,0.85833,0.69667,0.54667,0.45500],
                    [0.99000,0.87000,0.69333,0.58167,0.45667],
                    [1.00000,0.87000,0.68167,0.52167,0.42333],
                    [0.98333,0.86833,0.72833,0.54000,0.42667],
                    [1.00000,0.89833,0.72333,0.60167,0.46333],
                    [0.96000,0.87000,0.69333,0.54500,0.42167],
                    [0.99333,0.86000,0.66500,0.66500,0.44667],
                    [0.97667,0.81667,0.52167,0.52167,0.33833],
                    [0.97667,0.89667,0.73000,0.58333,0.48000],
                    [0.98000,0.85333,0.61333,0.42167,0.35500],
                    [0.95667,0.88333,0.61333,0.40667,0.35167]])
tda_min_color = '#539caf'


tda_ran = np.array([[0.99889,0.57833,0.45000,0.37167,0.35833],
	                [0.99833,0.57667,0.40500,0.36333,0.35333],
	                [0.95333,0.57833,0.45333,0.35833,0.36000],
                    [1.00000,0.58333,0.46500,0.38667,0.36833],
                    [1.00000,0.62667,0.45000,0.39500,0.37000],
                    [0.99000,0.61833,0.47333,0.38667,0.36833],
                    [1.00000,0.64000,0.47500,0.39333,0.37500],
                    [0.98333,0.65333,0.51167,0.41000,0.36667],
                    [1.00000,0.69167,0.50833,0.43500,0.38833],
                    [0.96000,0.69500,0.54667,0.44333,0.39500],
                    [0.99333,0.71000,0.52833,0.52833,0.39833],
                    [0.97667,0.63000,0.47833,0.47833,0.39333],
                    [0.97667,0.73833,0.57833,0.50000,0.46500],
                    [0.98000,0.73000,0.53833,0.43167,0.39333],
                    [0.95667,0.74833,0.52833,0.43167,0.38167]])

tda_ran_color = '#f97306'
# Create the plot object

_, ax = plt.subplots()
ax = plot_all(tda_max, ax, tda_max_color)
ax = plot_all(tda_min, ax, tda_min_color)
ax = plot_all(tda_ran, ax, tda_ran_color)

#ax.set_xticklabels(range(0, 5, 1), fontsize=14)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_title('TDA')
ax.set_xlabel('J')
ax.set_ylabel('Accuracy')
# Display legend
ax.legend(['Top-J', 'Bottom-J', 'Random-J'], loc='best')
plt.show()

ax.figure.savefig("tda.pdf", bbox_inches='tight')