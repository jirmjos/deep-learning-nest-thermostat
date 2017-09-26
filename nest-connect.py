
# coding: utf-8
# Simple fully connected network to learn cooling schedule for Nest Thermostat
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import nest
import datetime
import numpy as np
import os
import sys
import schedule
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation

client_id = '1f502532-26b1-4567-8c3a-c78ab184039a'
client_secret = 'RBsVN6rSgNOTL0e7gl1qUx40v'
access_token_cache_file = 'nest.json'
model_file = 'model.h5'
target_file = 'target.npy'


def input_fn(lr, last_change):
    now = datetime.datetime.now()
    diff = now - last_change
    print("Time since last update {}".format(diff.seconds))
    return np.array([[now.hour/24.0, now.minute/60.0,
                     diff.seconds/3600/24.0, (diff.seconds/60)%60/60.0,
                     lr.temperature/100.0, lr.humidity/100.0]]), np.array([[lr.target/100.0]])


def main():
    napi = nest.Nest(client_id=client_id, client_secret=client_secret, access_token_cache_file=access_token_cache_file)
    if napi.authorization_required:
        print("Please supply PIN")
        print(napi.authorize_url)
        if sys.version_info[0] < 3:
            pin = raw_input("PIN: ")
        else:
            pin = input("PIN: ")
        napi.request_token(pin)

    lr = napi.structures[0].thermostats[0]

    if os.path.isfile(model_file):
        print("Loading model")
        model = load_model(model_file)
    else:
        print("Generating model")
        model = Sequential()
        model.add(Dense(64, input_dim=6))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')

    if os.path.isfile(target_file):
        print("Loading saved target file")
        data = np.load(target_file)
    else:
        print("Generating new target file")
        data = np.array([datetime.datetime.now(), lr.target])
        np.save(target_file, data)

    x, y = input_fn(lr, data[0])

    print(x)
    print(y)

    if data[1] != lr.target:
        print("Target was updated {} {}".format(data[1], lr.target))
        # Setting changed, learn
        data = np.array([datetime.datetime.now(), lr.target])
        np.save(target_file, data)

    print("Training network")
    model.fit(x, y, nb_epoch=10, batch_size=1)
    new_target = model.predict(x) * 100.0
    print("New target {}".format(new_target))

    print(new_target)
    print(lr.temperature)
    print(lr.target)
    if new_target > lr.temperature and lr.target < lr.temperature:
        # turn off
        lr.target = new_target
    elif new_target < lr.temperature and lr.target > lr.temperature:
        # turn on
        lr.target = new_target
    else:
        print("Ignoring changes")

    model.save(model_file)

if __name__ == '__main__':
    main()
    schedule.every(1).minutes.do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
