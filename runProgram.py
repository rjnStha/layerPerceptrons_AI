import os
import signal

def run(value) :
	os.system('python3 perceptron.py '+value)

start = 10
stop = 200
step = 10

parameter = '100 bs'

for i in range(start,stop,step):
	if i == 0 : i = 1
	parameter = str(i)+' nn'
	run(parameter)
	#sound to notify completion of a test sample
	print('\a')


'''
if(paramerter_type == 'nn') : num_neurons_value = int(sys.argv[1])
elif(paramerter_type == 'av') :	activation_value = sys.argv[1]
elif(paramerter_type == 'bs') :	batch_size_value = int(sys.argv[1])
else(paramerter_type == 'lr') :	learning_rate_value = int(sys.argv[1])
'''