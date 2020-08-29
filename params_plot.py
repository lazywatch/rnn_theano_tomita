import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"


rnn_config_lib = {}
rnn_config_lib['SRN_sigmoid'] = {}
rnn_config_lib['SRN_sigmoid']['name'] = 'Elman-Sig'
rnn_config_lib['SRN_sigmoid']['weights'] = ['U','V','B']
rnn_config_lib['SRN_sigmoid']['nhid'] = [64, 64, 64, 64, 128, 128, 64]

rnn_config_lib['SRN_tanh'] = {}
rnn_config_lib['SRN_tanh']['name'] = 'Elman-Tanh'
rnn_config_lib['SRN_tanh']['weights'] = ['U','V','B']
rnn_config_lib['SRN_tanh']['nhid'] = [64, 64, 64, 64, 128, 128, 64]


rnn_config_lib['O2_sigmoid'] = {}
rnn_config_lib['O2_sigmoid']['name'] = '2-RNN-Sig'
rnn_config_lib['O2_sigmoid']['weights'] = ['W']
rnn_config_lib['O2_sigmoid']['nhid'] = [46, 46, 46, 46, 91, 91, 46]

rnn_config_lib['O2_tanh'] = {}
rnn_config_lib['O2_tanh']['name'] = '2-RNN-Tanh'
rnn_config_lib['O2_tanh']['weights'] = ['W']
rnn_config_lib['O2_tanh']['nhid'] = [46, 46, 46, 46, 91, 91, 46]


rnn_config_lib['UNI_sigmoid'] = {}
rnn_config_lib['UNI_sigmoid']['name'] = 'UNI-Sig'
rnn_config_lib['UNI_sigmoid']['weights'] = ['W','U','V','B']
rnn_config_lib['UNI_sigmoid']['nhid'] = [37, 37, 37, 37, 73, 73, 37]

rnn_config_lib['UNI_tanh'] = {}
rnn_config_lib['UNI_tanh']['name'] = 'UNI-Tanh'
rnn_config_lib['UNI_tanh']['weights'] = ['W','U','V','B']
rnn_config_lib['UNI_tanh']['nhid'] = [37, 37, 37, 37, 73, 73, 37]


rnn_config_lib['MI_tanh'] = {}
rnn_config_lib['MI_tanh']['name'] = 'MI'
rnn_config_lib['MI_tanh']['weights'] = ['U','V','B','alpha','beta1','beta2']
rnn_config_lib['MI_tanh']['nhid'] = [62, 62, 62, 62, 126, 126, 62]


rnn_names = ['SRN_sigmoid','O2_sigmoid','UNI_sigmoid','MI_tanh']
num_grammars = 7
seed = 1

fig, ax = plt.subplots(nrows=len(rnn_names), ncols=num_grammars, figsize=(35, 20))
# fig.suptitle('The L2-norm of updated weights during training', fontsize=20)
for row, m in zip(ax, rnn_names):
    for col, g in zip(row, range(1, num_grammars + 1)):
        params_log_filename = './params/tomita/g' + str(g) + '_' + m + '_h' \
                              + str(rnn_config_lib[m]['nhid'][g - 1]) \
                              + '_seed' + str(seed) + '_params_log.npz'

        params_log_file = np.load(params_log_filename)
        params_log = params_log_file['log']
        assert params_log.shape[1] == len(rnn_config_lib[m]['weights'])
        # fig = plt.figure(figsize=(6, 4))
        for j in range(len(rnn_config_lib[m]['weights'])):
            col.plot(params_log[:, j], label=rnn_config_lib[m]['weights'][j])

        # plt.title(rnn_config_lib[m]['name'] + " Grammar " + str(g), fontsize=16)
        # plt.ylabel('L2-norm of weights', fontsize=16)
        # plt.xlabel('epoch', fontsize=16)
        col.legend(loc='best', fontsize=16)
        # plt.show()

plt.savefig("weights_change.pdf",
            bbox_inches='tight')
plt.show()


