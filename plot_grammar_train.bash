#python grammar_plot.py --rnn SRN_sigmoid --nhid 50
#python grammar_plot.py --rnn SRN_tanh --nhid 50
#python grammar_plot.py --rnn O2_sigmoid --nhid 50
#python grammar_plot.py --rnn O2_tanh --nhid 50
#python grammar_plot.py --rnn GRU_tanh --nhid 50
#python grammar_plot.py --rnn LSTM_tanh --nhid 50

python grammar_plot.py --rnn SRN_sigmoid --nhid 50 --len 200
python grammar_plot.py --rnn SRN_tanh --nhid 50 --len 200
python grammar_plot.py --rnn O2_sigmoid --nhid 50 --len 200
python grammar_plot.py --rnn O2_tanh --nhid 50 --len 200
python grammar_plot.py --rnn GRU_tanh --nhid 50 --len 200
python grammar_plot.py --rnn LSTM_tanh --nhid 50 --len 200