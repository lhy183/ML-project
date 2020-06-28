from __future__ import print_function, division
from warnings import warn, filterwarnings

import random
import sys
import numpy as np
import time
import json

from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

from nilmtk import DataSet
import metrics
from gen import opends, gen_batch
from model import create_model

allowed_key_names = ['fridge','microwave','dish_washer','kettle','washing_machine']

def normalize(data, mmax, mean, std):
	return data / mmax

def denormalize(data, mmax, mean, std):
	return data * mmax

def experiment(key_name, start_e, end_e):
	
	if (key_name not in allowed_key_names):
	conf_filename = "appconf/{}.json".format(key_name)
	with open(conf_filename) as data_file:
		conf = json.load(data_file)

	input_window = conf['lookback']
	threshold = conf['on_threshold']
	mamax = 5000
	memax = conf['memax']
	mean = conf['mean']
	std = conf['std']
	train_buildings = conf['train_buildings']
	test_building = conf['test_building']
	on_threshold = conf['on_threshold']
	meter_key = conf['nilmtk_key']
	save_path = conf['save_path']




	X_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
	X_train = normalize(X_train, mamax, mean, std)
	y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
	y_train = normalize(y_train, memax, mean, std)
	model = create_model(input_window)

	if start_e > 0:
		model = load_model(save_path+"CHECKPOINT-{}-{}epochs.hdf5".format(key_name, start_e))

	if end_e > start_e:
		filepath = save_path+"CHECKPOINT-"+key_name+"-{epoch:01d}epochs.hdf5"
		checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
		history = model.fit(X_train, y_train, batch_size=128, epochs=(end_e - start_e), shuffle=True, initial_epoch=start_e, callbacks=[checkpoint])
		losses = history.history['loss']

		model.save("{}CHECKPOINT-{}-{}epochs.hdf5".format(save_path, key_name, end_e),model)


		try:
			a = np.loadtxt("{}losses.csv".format(save_path))
			losses = np.append(a,losses)
		except:
			pass
		np.savetxt("{}losses.csv".format(save_path), losses, delimiter=",")


	mains, meter = opends(test_building, key_name)
	X_test = normalize(mains, mamax, mean, std)
	y_test = meter


	X_batch, Y_batch = gen_batch(X_test, y_test, len(X_test)-input_window, 0, input_window)
	pred = model.predict(X_batch)
	pred = denormalize(pred, memax, mean, std)
	pred[pred<0] = 0
	pred = np.transpose(pred)[0]

	np.save("{}pred-{}-epochs{}".format(save_path, key_name, end_e), pred)

	rpaf = metrics.recall_precision_accuracy_f1(pred, Y_batch, threshold)
	rete = metrics.relative_error_total_energy(pred, Y_batch)
	mae = metrics.mean_absolute_error(pred, Y_batch)


	res_out = open("{}results-pred-{}-{}epochs".format(save_path, key_name, end_e), 'w')
	for r in rpaf:
		res_out.write(str(r))
		res_out.write(',')
	res_out.write(str(rete))
	res_out.write(',')
	res_out.write(str(mae))
	res_out.close()


if __name__ == "__main__":
	if len(sys.argv) == 1 or sys.argv[1] == "":
		exit()

	key_name = sys.argv[1]
	experiment(key_name, 0, 3)
