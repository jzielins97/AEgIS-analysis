my_dict = {'acq_0':{'Timestamp':{'value':10,'something':123}},'acq_1':{'Timestamp':{'value':15,'something':432}}}

for i, (key, entry) in enumerate(my_dict.items()):
    print(i,entry)