import argparse
import os, sys
import time
import numpy as np

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import kmc2
import DFA
from utils import dfa_perf_measure

parser = argparse.ArgumentParser(description='DFA extraction with k-means')

parser.add_argument('--rnn', type=str, default='LSTM', help='which rnn model')
parser.add_argument('--act', type=str, default='tanh', help='which rnn model')
#parser.add_argument('--hseed', type=int, default=1234, help='random seed for initializing hidden layer')
parser.add_argument('--ninp', type=int, default=2, help='input dimension')
parser.add_argument('--nhid', type=int, default=31, help='hidden dimension')
parser.add_argument('--seed', type=int, default=1, help='random seed for initialize weights')

parser.add_argument('--data', type=str, default='g1', help='index of Tomita grammars')
parser.add_argument('--k_min', type=int, default=2, help='minimum k, minimum number of states')
parser.add_argument('--k_max', type=int, default=20, help='maximum k, maximum number of states')
parser.add_argument('--kseed', type=int, default=1, help='random seed for initialize k-means')

args = parser.parse_args()

np.random.seed(args.kseed)

def test_dfa(x, y, dfa, save_file = None):
    y_pred = np.zeros((len(x),), dtype='int32')
    for x_id in range(len(x)):
        #input_seq = [str(seq) for seq in x[seq_id, :np.where(x[seq_id]==2)[0][0]]]
        input_seq = [str(seq) for seq in x[x_id]]
        input_seq = ''.join(input_seq)
        dfa.reset()
        dfa.input_sequence(input_seq)
        if dfa.status():
            y_pred[x_id] = 1
        dfa.reset()

    if y is not None:
        (precision, recall, accuracy, f1) = dfa_perf_measure(y_true=y, y_pred=y_pred, use_self=False)
        print("Precision: %.4f Recall: %.4f Accuracy: %.4f F1: %.4f" % (precision, recall, accuracy, f1))
        if f1 == 1.0:
            np.savez(save_file, transitions=dfa.transitions, states=dfa.states,
                     start=dfa.start, accepts=dfa.accepts, alphabet=dfa.alphabet)

        return f1
    else:
        return y_pred

data_type = 'float64'

alphabet = ['0','1']


print('G:%s, model:%s %s, load all data and stored hidden activations' %(args.data, args.rnn, args.act))
save_dir = ''.join(('./params/tomita/', args.data, '_',
                    args.rnn, '_', args.act, '_h', str(args.nhid),
                    '_seed', str(args.seed)))

train_val_test_file = np.load(''.join(('./data/Tomita/',
                               args.data,
                               '_train_val_test_data_lstar.npz')))

x = train_val_test_file['test_x']
m = train_val_test_file['test_m']
y = train_val_test_file['test_y']

hinit_file = np.load(''.join((save_dir, '_hinit.npz')))
h_log_file = np.load(''.join((save_dir, '_h_log.npz')))

h_init = hinit_file['hinit']
if args.rnn == "LSTM":
    h_init = np.tile(h_init, (1, 2)).squeeze()

hstates = h_log_file['log']


n_pos = np.where(y == 1)[0].shape[0]
n_sample = x.shape[0]
str_len = x.shape[1]


x_list = []
h_list = np.array([], dtype='float32').reshape(0,h_init.shape[0])

for x_id in range(n_sample):
    if np.where(m[x_id] == 0)[0].shape[0] > 0:
        col_end_idx = np.where(m[x_id] == 0)[0][0]
    else:
        col_end_idx = m.shape[-1]
    x_list.append(x[x_id, :col_end_idx])
    #h_tmp = np.concatenate((h_init[None,:], hstates[row_id,:col_id]),axis=0)
    h_list = np.concatenate((h_list, hstates[x_id,:col_end_idx,:]),axis=0)

h_list = StandardScaler().fit_transform(h_list)
accuracy_log = np.zeros((args.k_max-args.k_min,),dtype=data_type)
#accuracy_gt_log = np.zeros((args.k_max-args.k_min,),dtype=data_type)

h_all = np.concatenate((h_list, np.tile(h_init, (n_sample, 1))),axis=0)

for n_states in range(args.k_min,args.k_max):
    print('\n')
    print('--------------------------------------------------------------------')
    print('Begin DFA extraction with n_state:%d' % (n_states))
    states = [str(ind) for ind in range(1, n_states + 1)]

    start_clustering_time = time.time()
    #cluster_model = cluster.MiniBatchKMeans(n_clusters=n_states, random_state=1)
    #cluster_model = cluster.KMeans(n_clusters=n_states, random_state=1)
    #cluster_model = cluster.Birch(n_clusters=n_states)
    #cluster_model = cluster.SpectralClustering(n_clusters=n_states,
    #                                          eigen_solver='arpack',
    #                                          affinity="nearest_neighbors",
    #                                          random_state=1)



    seeding = kmc2.kmc2(X=h_all, k=n_states, chain_length=200, afkmc2=True,
                        weights=None, random_state=np.random.RandomState(0))
    cluster_model = cluster.MiniBatchKMeans(n_clusters=n_states, init=seeding)
    cluster_model.fit(h_all)
    print('Done fitting takes time: %.4f' % (time.time() - start_clustering_time))
    sys.stdout.flush()

    h_pred = cluster_model.predict(h_list) + 1
    #score = []
    #for i in range(random_times):
    #    random_idx = np.random.choice(h_pred.shape[0], int(np.floor(h_pred.shape[0]/random_times)))
    #    score.append(metrics.silhouette_score(h_all[random_idx], h_pred[random_idx], metric='euclidean'))
    #print("K : %d, silhouette : %.4f" %(n_states, np.mean(score)))

    transitions = {}

    for idx in range(n_states):
        transitions[str(idx + 1)] = {}

    start_state = cluster_model.predict(h_init.reshape(1,-1)) + 1# h_pred[0]
    end_states = np.zeros(n_sample, 'int32')
    transit_states_cnt = np.zeros((n_states, args.ninp, n_states), dtype='int')

    #h_pred_list = []
    seq_start_idx = 0
    for x_id in range(n_sample):
        if np.where(m[x_id] == 0)[0].shape[0] > 0:
            col_end_idx = np.where(m[x_id] == 0)[0][0]
        else:
            col_end_idx = m.shape[-1]

        h_pred_per_sample = np.concatenate((start_state, h_pred[seq_start_idx:seq_start_idx + col_end_idx]))

        #h_pred_list.append(h_pred_per_sample)
        for seq_id in range(col_end_idx):
            current_state = h_pred_per_sample[seq_id]
            current_input = x_list[x_id][seq_id]
            next_state = h_pred_per_sample[seq_id+1]
            transitions[str(current_state)][str(current_input)] = str(next_state)
            transit_states_cnt[current_state-1, current_input, next_state-1] += 1

        assert next_state == h_pred_per_sample[-1]
        seq_start_idx = seq_start_idx + col_end_idx
        end_states[x_id] = next_state

    unique_end_states = np.unique(end_states[np.where(y == 1)[0]])
    accepts = [str(ind) for ind in unique_end_states]
    #unique_end_states, unique_end_state_counts = np.unique(end_states, return_counts=True)
    #accepts = [str(ind) for ind in unique_end_states]

    for state_idx in range(n_states):
        states_log = np.zeros(args.ninp)
        for input_idx in range(args.ninp):
            if (np.sum(transit_states_cnt[state_idx,input_idx,:] == 0) == n_states):
                transitions[str(state_idx+1)][str(input_idx)] = str(state_idx+1)
            elif (np.sum(transit_states_cnt[state_idx,input_idx,:] == 0) < n_states-1):
                idx = np.where(transit_states_cnt[state_idx,input_idx,:]>0)[0]
                multi_states = transit_states_cnt[state_idx,input_idx,idx]
                idx = idx[np.argmax(multi_states)]
                transitions[str(state_idx+1)][str(input_idx)] = str(idx+1)
            states_log[input_idx] = int(transitions[str(state_idx + 1)][str(input_idx)])
        # all goes to the same state, this is an absorbing state
        if np.unique(states_log).shape[0] == 1:
            if str(int(np.unique(states_log)[0])) in accepts:
                accepts.remove(transitions[str(state_idx + 1)][str(input_idx)])

    #print(transit_states_cnt)
    #print(transitions)

    start = str(start_state[0])

    delta = (lambda s, a: transitions[s][a])

    d0 = DFA.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
    d0.reset()
    print("==Minimized===")
    d0.minimize()
    d0.pretty_print()
    d0.reset()

    np.savez(''.join((save_dir, '_dfa_config.npz')),
             transitions=transitions, states=states,
             start=start, accepts=accepts, alphabet=alphabet)

    accuracy_log[n_states-args.k_min] = test_dfa(x=x_list, y=y, dfa=d0,
                                                 save_file=''.join((save_dir, '_gt_dfa_config.npz')))

    sys.stdout.flush()
    
    #if args.long_string:
        #print('loading valid, invalid samples with long sequence')
    #    npzfile = np.load(''.join((save_dir,'/pos_set_long.npz')))
    #    val_x = npzfile['x']
    #    val_mask = npzfile['mask']

    #    npzfile = np.load(''.join((save_dir,'/neg_set_long.npz')))
    #    inval_x = npzfile['x']
    #    inval_mask = npzfile['mask']

        #accuracy_long_log[n_states-args.k_min] = test_dfa(val_x, inval_x, d0)
#print(accuracy_gt_log)

print(accuracy_log)
print("\n")

extract_f1_log_filename = ''.join(('./params/tomita/', args.data,
                                   '_seed', str(args.seed), '_f1_log.npz'))
if os.path.exists(extract_f1_log_filename):
    extract_f1_log_file = np.load(extract_f1_log_filename)
    extract_f1_log = extract_f1_log_file['extract_f1_log']
else:
    extract_f1_log = np.zeros((9,accuracy_log.shape[0]))

if (args.rnn == "SRN"):
    if (args.act == "sigmoid"):
        extract_f1_log[0] = accuracy_log
    elif (args.act == "tanh"):
        extract_f1_log[1] = accuracy_log
elif (args.rnn == "O2"):
    if (args.act == "sigmoid"):
        extract_f1_log[2] = accuracy_log
    elif (args.act == "tanh"):
        extract_f1_log[3] = accuracy_log
elif (args.rnn == "UNI"):
    if (args.act == "sigmoid"):
        extract_f1_log[4] = accuracy_log
    elif (args.act == "tanh"):
        extract_f1_log[5] = accuracy_log
elif (args.rnn == "MI"):
    extract_f1_log[6] = accuracy_log
elif (args.rnn == "LSTM"):
    extract_f1_log[7] = accuracy_log
elif (args.rnn == "GRU"):
    extract_f1_log[8] = accuracy_log

print(extract_f1_log)
np.savez(extract_f1_log_filename, extract_f1_log=extract_f1_log)





