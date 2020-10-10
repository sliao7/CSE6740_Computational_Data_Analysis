from matplotlib import pyplot as plt
k = [3,5,10,16,32]
pictures = ['flower','beach','football']


# kmedoids results
flower_times = [28,89,46,138,207]
flower_iters = [6,19,6,12,14]
flower_clusters = [3,5,8,14,18]

beach_times = [77,50,127,165,160]
beach_iters = [15,9,19,15,9]
beach_clusters = [3,5,8,14,21]

football_times = [441,458,619,695,810]
football_iters = [9,13,16,14,10]
football_clusters = [3,5,9,14,24]

plt.plot(k,flower_times,'o--', label = 'flower')
plt.plot(k,beach_times, '*--', label = 'beach')
plt.plot(k,football_times, 'o--', label = 'football')
plt.title('k-medoids convergence times')
plt.xlabel('k')
plt.ylabel('running time in seconds')
plt.legend()
plt.savefig('results/kmedoids_times.png')   # save the figure to file
plt.close()    # close the figure window

plt.plot(k,flower_iters,'o--', label = 'flower')
plt.plot(k,beach_iters, '*--', label = 'beach')
plt.plot(k,football_iters, 'o--', label = 'football')
plt.title('k-medoids number of iterations')
plt.xlabel('k')
plt.ylabel('number of iterations')
plt.legend()
plt.savefig('results/kmedoids_iters.png')   # save the figure to file
plt.close()    # close the figure window

# plt.plot(k,flower_clusters,label = 'flower')
# plt.plot(k,beach_clusters, label = 'beach')
# plt.plot(k,football_clusters, label = 'football')
# plt.title('k-medoids number of resulting clusters')
# plt.xlabel('k')
# plt.ylabel('number of resulting clusters')
# plt.legend()
# fig.savefig('results/kmedoids_iters.png')   # save the figure to file
# plt.close(fig)    # close the figure window

# plt.plot(k,beach_times,'o--',label = 'k-medoids')

#kmeans results
flower_times = [.14,.32,.57,1.43,5.22]
flower_iters = [25,48,65,113,297]
flower_clusters = [3,5,8,14,22]

beach_times = [.08,.19,.81,1.42,2.39]
beach_iters = [11,26,88,127,123]
beach_clusters = [3,5,9,12,25]

football_times = [.39,.91,1.55,3.46,5.24]
football_iters = [18,35,48,68,72]
football_clusters = [3,5,8,16,23]

# plt.plot(k,beach_times,'*--',label = 'kmeans')
# plt.title('Running time: k-medoids VS kmeans')
# plt.xlabel('k')
# plt.ylabel('running time in seconds')
# plt.legend()
# plt.show()

plt.plot(k,flower_times,'o--', label = 'flower')
plt.plot(k,beach_times, '*--', label = 'beach')
plt.plot(k,football_times, 'o--', label = 'football')
plt.title('k-means convergence times')
plt.xlabel('k')
plt.ylabel('running time in seconds')
plt.legend()
plt.savefig('results/kmeans_times.png')   # save the figure to file
plt.close()    # close the figure window

plt.plot(k,flower_iters,'o--', label = 'flower')
plt.plot(k,beach_iters, '*--', label = 'beach')
plt.plot(k,football_iters, 'o--', label = 'football')
plt.title('k-means number of iterations')
plt.xlabel('k')
plt.ylabel('number of iterations')
plt.legend()
plt.savefig('results/kmeans_iters.png')   # save the figure to file
plt.close()    # close the figure window

# plt.plot(k,flower_clusters,label = 'flower')
# plt.plot(k,beach_clusters, label = 'beach')
# plt.plot(k,football_clusters, label = 'football')
# plt.title('k-means number of resulting clusters')
# plt.legend()
# plt.show()




