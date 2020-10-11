V = 'secret, offer, low, price, valued, customer, today, dollar, million, sports, is, for, play, healthy, pizza'
V = V.split(', ')

spam1 = ['million dollar offer', 'secret offer today', 'secret is secret']
spam = [s.split(' ') for s in spam1]

nonspam1 = ['low price for valued customer', 'play secret sports today', 'sports is healthy', 'low price pizza']
nonspam = [s.split(' ') for s in nonspam1]

def feature_vec(data,data1):
    for i in range(len(data)):
        email = data[i]
        vector = []
        for word in V:
            vector.append(email.count(word))

        print('&', data1[i],'&',  vector, '\\' + '\\')
feature_vec(spam,spam1)
feature_vec(nonspam,nonspam1)