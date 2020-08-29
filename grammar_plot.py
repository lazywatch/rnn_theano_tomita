import matplotlib
#print(matplotlib.get_cachedir())
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import scipy.stats
import argparse
plt.rcParams["font.family"] = "Times New Roman"



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
    data_stats = [mean_confidence_interval(data[:,i]) for i in range(data.shape[1])]
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
    

parser = argparse.ArgumentParser(description='RNN trained on Tomita grammars')
parser.add_argument('--rnn', type=str, default='O2_sigmoid', help='rnn model')
parser.add_argument('--nhid', type=int, default=10, help='hidden dimension')
parser.add_argument('--len', type=int, default=-1, help='len to zoom in')

args = parser.parse_args()


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

n_trials = 10
i = 0
max_len = 5000

g1_loss_log = np.zeros((n_trials, max_len))
g2_loss_log = np.zeros((n_trials, max_len))
g3_loss_log = np.zeros((n_trials, max_len))
g4_loss_log = np.zeros((n_trials, max_len))
g5_loss_log = np.zeros((n_trials, max_len))
g6_loss_log = np.zeros((n_trials, max_len))
g7_loss_log = np.zeros((n_trials, max_len))


for i in range(n_trials):
    g1_file = ''.join(('./params/tomita/g1_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g1_loss_log1 = np.load(g1_file)['log']

    g2_file = ''.join(('./params/tomita/g2_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g2_loss_log1 = np.load(g2_file)['log']

    g3_file = ''.join(('./params/tomita/g3_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g3_loss_log1 = np.load(g3_file)['log']

    g4_file = ''.join(('./params/tomita/g4_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g4_loss_log1 = np.load(g4_file)['log']

    g5_file = ''.join(('./params/tomita/g5_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g5_loss_log1 = np.load(g5_file)['log']

    g6_file = ''.join(('./params/tomita/g6_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g6_loss_log1 = np.load(g6_file)['log']

    g7_file = ''.join(('./params/tomita/g7_',
                       args.rnn, '_h', str(args.nhid),
                       '_seed', str(i + 1), '_train_loss_log.npz'))
    g7_loss_log1 = np.load(g7_file)['log']

    max_len = np.max([max_len, len(g1_loss_log1),
                      len(g2_loss_log1), len(g3_loss_log1),
                      len(g4_loss_log1), len(g5_loss_log1),
                      len(g6_loss_log1), len(g7_loss_log1)])

    g1_loss_log[i, :g1_loss_log1.shape[0]] = g1_loss_log1
    g2_loss_log[i, :g2_loss_log1.shape[0]] = g2_loss_log1
    g3_loss_log[i, :g3_loss_log1.shape[0]] = g3_loss_log1
    g4_loss_log[i, :g4_loss_log1.shape[0]] = g4_loss_log1
    g5_loss_log[i, :g5_loss_log1.shape[0]] = g5_loss_log1
    g6_loss_log[i, :g6_loss_log1.shape[0]] = g6_loss_log1
    g7_loss_log[i, :g7_loss_log1.shape[0]] = g7_loss_log1

g1_loss_log = g1_loss_log[:,:max_len]
g2_loss_log = g2_loss_log[:,:max_len]
g3_loss_log = g3_loss_log[:,:max_len]
g4_loss_log = g4_loss_log[:,:max_len]
g5_loss_log = g5_loss_log[:,:max_len]
g6_loss_log = g6_loss_log[:,:max_len]
g7_loss_log = g7_loss_log[:,:max_len]
#color_list = ['skyblue', "green", "red", "cyan", "magenta",
#              "yellow", "black", "olive", "darkorange"]

#g1_color = '#539caf'
#g3_color = '#be0119'
#g5_color = '#6f828a'

color_dict = {}
color_dict['g1'] = 'skyblue'
color_dict['g2'] = 'green'
color_dict['g3'] = 'red'
color_dict['g4'] = 'cyan'
color_dict['g5'] = 'magenta'
color_dict['g6'] = 'darkorange'
color_dict['g7'] = 'black'

# Create the plot object

_, ax = plt.subplots()
if args.len > 0:
    ax = plot_all(g1_loss_log[:,:args.len], ax, color_dict['g1'])
    #ax = plot_all(g2_loss_log[:,:args.len], ax, color_dict['g2'])
    ax = plot_all(g3_loss_log[:,:args.len], ax, color_dict['g3'])
    #ax = plot_all(g4_loss_log[:,:args.len], ax, color_dict['g4'])
    ax = plot_all(g5_loss_log[:,:args.len], ax, color_dict['g5'])
    #ax = plot_all(g6_loss_log[:,:args.len], ax, color_dict['g6'])
    #ax = plot_all(g7_loss_log[:,:args.len], ax, color_dict['g7'])
else:
    ax = plot_all(g1_loss_log, ax, color_dict['g1'])
    #ax = plot_all(g2_loss_log, ax, color_dict['g2'])
    ax = plot_all(g3_loss_log, ax, color_dict['g3'])
    #ax = plot_all(g4_loss_log, ax, color_dict['g4'])
    ax = plot_all(g5_loss_log, ax, color_dict['g5'])
    #ax = plot_all(g6_loss_log, ax, color_dict['g6'])
    #ax = plot_all(g7_loss_log, ax, color_dict['g7'])
#ax.set_xticklabels(range(0, 5, 1), fontsize=14)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_title('TDA')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# Display legend
#ax.legend(['G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7'], loc='best')
ax.legend(['G-1', 'G-3', 'G-5'], loc='best')
#plt.show()

if args.len > 0:
    ax.figure.savefig(''.join((args.rnn, '_h', str(args.nhid), '_zoom.pdf')),
                      bbox_inches='tight')
else:
    ax.figure.savefig(''.join((args.rnn,'_h', str(args.nhid) ,'.pdf')),
                      bbox_inches='tight')