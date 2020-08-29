import matplotlib.pyplot as plt

import os
import sys
import pandas as pd
import numpy as np
#csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

color_list = ['skyblue', "green", "red", "cyan", "magenta",
              "yellow", "black", "olive", "darkorange"]
model_list = ["Elman-Sig", "Elman-Tanh", "2-RNN-Sig", "2-RNN-Tanh",
              "UNI-Sig", "UNI-Tanh", "MI", "LSTM", "GRU"]

k = np.arange(18)

num_grammars = 7
num_trials = 10
num_models = len(model_list)
num_k = k.shape[0]

extract_log_all = np.zeros((num_grammars, num_trials, num_models, num_k))

for grammar_id in range(1, num_grammars + 1):
    for seed in range(1, 11):
        extract_f1_log_filename = ''.join(('./params/tomita/g', str(grammar_id),
                                           '_seed', str(seed), '_f1_log.npz'))
        if os.path.exists(extract_f1_log_filename):
            extract_log_file = np.load(extract_f1_log_filename)
            extract_log_all[grammar_id - 1, seed - 1] = extract_log_file['extract_f1_log']
        else:
            print("no such file " + extract_f1_log_filename)

    y = np.mean(extract_log_all, axis=1)
    yerr = np.var(extract_log_all, axis=1)

    fig = plt.figure(figsize=(12, 6))
    for model_id in range(num_models):
        plt.errorbar(k, y[grammar_id - 1, model_id], yerr=yerr[grammar_id - 1, model_id],
                     color=color_list[model_id], linewidth=1, fmt='-o', uplims=True, lolims=True,
                     label=model_list[model_id], alpha=0.6)

    plt.ylabel('F1 score', fontsize=16)
    #plt.title('Performance of extracted DFA for grammar ' + str(grammar_id),
    #          fontsize=16)
    plt.xticks(k, [str(i) for i in range(2, 20)], fontsize=16)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.savefig("grammar_" + str(grammar_id) + "_acc.pdf",
                bbox_inches='tight')
    plt.show()

success_log = np.zeros((num_grammars, num_models))
for grammar_id in range(1, num_grammars + 1):
    for model_id in range(num_models):
        success_log[grammar_id - 1, model_id] = np.where(
            extract_log_all[grammar_id - 1, :, model_id, :] == 1.0
        )[0].shape[0]
        success_log[grammar_id - 1, model_id] = success_log[grammar_id - 1, model_id] / 180.0

ind = np.arange(num_grammars)
width = 0.1

fig = plt.figure(figsize=(30, 10))
for i, m, c in zip(np.arange(num_models), model_list, color_list):
    plt.bar(ind + i*width, success_log[:,i], width, label=m, color=c)

plt.ylabel('Success rate', fontsize=20)


plt.xticks(ind + width*4, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'),
           fontsize=16)
plt.yticks(np.arange(0,1,0.1), ('0%', '10%', '20%', '30%',
                                '40%', '50%', '60%', '70%',
                                '80%', '90%'),
           fontsize=16)

plt.legend(loc='best', fontsize=16)
plt.savefig("success_rate.pdf",bbox_inches='tight')
plt.show()

