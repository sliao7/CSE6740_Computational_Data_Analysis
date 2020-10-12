# CSE6740 homework4 problem3
# This file is used to calculate vectors for messages and generate latex codes 

import numpy as np
V = 'secret, offer, low, price, valued, customer, today, dollar, million, sports, is, for, play, healthy, pizza'
V = V.split(', ')

spam1 = ['million dollar offer', 'secret offer today', 'secret is secret']
spam = [s.split(' ') for s in spam1]

nonspam1 = ['low price for valued customer', 'play secret sports today', 'sports is healthy', 'low price pizza']
nonspam = [s.split(' ') for s in nonspam1]

k = 1
def feature_vec(data,data1):
    vectors = []
    for i in range(len(data)):
        email = data[i]
        vector = []
        for word in V:
            vector.append(email.count(word))

        # print('&', data1[i],'&',  vector, '\\' + '\\')
        # print(i,[(j+1, vector[j]) for j in range(len(vector)) if vector[j] != 0])
        # print('')
        vectors.append(vector)
    return vectors

data = feature_vec(spam,spam1) + feature_vec(nonspam,nonspam1)
# for i in range(1,8):
#     print('$ i = ', i, ' : ' ,end = '')
#     for k,x in enumerate(data[i-1]):
#         if x != 0:
#             print('x^' + '{(' +  str(i) + ')}'  + '_' + '{' + str(k+1) + '}'  + ' = '  + str(x) + ', ', end = '')
#     print('$' + '\\' + '\\')
#     print('')

data = np.array(data)
spam_sum = sum(data[:3,:])
nonspam_sum = sum(data[3:,:])





