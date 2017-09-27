import pandas as pd



with open ("../data/assign2_train_data.txt", "r") as myfile:
    data = myfile.readlines()

data = pd.read_csv("../data/assign2_train_data.txt", names=["i_data", "date"])


print "data: ", data
print "Date: ", data["date"]
