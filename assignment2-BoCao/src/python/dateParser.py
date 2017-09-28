import pandas as pd
import datetime

train_data_pd = pd.read_csv("../data/assign2_train_data.txt", names=["index", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRation", "Occupancy"])

def normalize_date(date_t):
    time_t = []
    for ii in range(len(date_t)):
        time_value = 0
        digits = date_t[ii].split(" ")[1].split(":")
        time_value += int(digits[0]) * 3600
        time_value += int(digits[1]) * 60
        time_value += int(digits[2])
        print "date_t[ii]: ", date_t[ii], "time_value: ", float(time_value) / 86400

#print "data: ", data
date_pd = normalize_date(train_data_pd["date"])
