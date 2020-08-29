import matplotlib
#print(matplotlib.get_cachedir())
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.92):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    b = np.array([m, m-h, m+h])
    return b.clip(0)

def mean_max_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    m, min, max = np.mean(a), np.min(a), np.max(a)
    return [m, min, max]

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
    data_stats = [mean_max_interval(data[:,i]) for i in range(data.shape[1])]
    data_stats = np.array(data_stats).T
    predict_mean_ci_low = data_stats[1,:]#np.min(data,axis=0)
    predict_mean_ci_upp = data_stats[2,:]#np.max(data,axis=0)
    # Data for regions where we want to shade to indicate the intervals has
    # to be sorted by the x axis to display correctly
    CI_df = pd.DataFrame(columns = ['x_data', 'low_CI', 'upper_CI'])
    CI_df['x_data'] = np.arange(start=0,stop=data_stats.shape[1],step=1)
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
    ax.plot(np.arange(start=0,stop=data_stats.shape[1],step=1), data_stats[0,:],
            lw = 1, color = color_code, alpha = 1)#, label = 'Mean')
    # Shade the confidence interval
    ax.fill_between(CI_df['x_data'], CI_df['low_CI'], CI_df['upper_CI'],
                    color = color_code, alpha = 0.4)#, label = '95% CI')
    # Label the axes and provide a title
    return ax
    

# Set some parameters to apply to all plots. These can be overridden
# in each plot if desired

# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (4, 2))#(14, 7))
# Font size to 14
matplotlib.rc('font', size = 10)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

n_trials = 20
rnn_type = 'gru_h80'
i = 0
g1_file = ''.join(('../gram1/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
g1_loss_log1 = np.load(g1_file)['train_loss_log']
g3_file = ''.join(('../gram3/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
g3_loss_log1 = np.load(g3_file)['train_loss_log']
g5_file = ''.join(('../gram5/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
g5_loss_log1 = np.load(g5_file)['train_loss_log']


max_len = np.max([len(g1_loss_log1), len(g3_loss_log1), len(g5_loss_log1)])

g1_loss_log = np.zeros((n_trials, max_len))
g3_loss_log = np.zeros((n_trials, max_len))
g5_loss_log = np.zeros((n_trials, max_len))


for i in range(n_trials):
    g1_file = ''.join(('../gram1/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
    tmp = np.load(g1_file)['train_loss_log']
    g1_loss_log[i,:tmp.shape[0]] = tmp

    g3_file = ''.join(('../gram3/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
    tmp = np.load(g3_file)['train_loss_log']
    g3_loss_log[i,:tmp.shape[0]] = tmp

    g5_file = ''.join(('../gram5/',rnn_type,'_seed',str(i+1),'_train_loss_log.npz'))
    tmp = np.load(g5_file)['train_loss_log']
    g5_loss_log[i,:tmp.shape[0]] = tmp


g1_color = '#539caf'
g3_color = '#be0119'
g5_color = '#6f828a'
# Create the plot object
zoom_len = 25
_, ax = plt.subplots()
ax = plot_all(g1_loss_log[:,:zoom_len], ax, g1_color)
ax = plot_all(g5_loss_log[:,:zoom_len], ax, g5_color)
ax = plot_all(g3_loss_log[:,:zoom_len], ax, g3_color)

#ax.set_xticklabels(range(0, 5, 1), fontsize=14)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_title('TDA')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# Display legend
#ax.legend(['Grammar 1', 'Grammar 3', 'Grammar 5'], loc='best')
plt.show()

ax.figure.savefig(''.join((rnn_type,'_zoom.pdf')), bbox_inches='tight')