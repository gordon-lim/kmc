import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time

# load dataset
# [ number of packets sent per second           size of packet ]
# anomaly (DDOS attempts) will have lots of big packets sent in a short amount of time

# L2 norm aka Euclidean distance function
def euclidean(a, b):
    return np.linalg.norm(a-b)

def kmeans(k, dataset):
    # list to store past centroid
    history_centroids = []
    # get the number of rows (instances) and columns (features) from
    # the dataset
    num_instances, num_features = dataset.shape
    # randomly pick k points from our datasets to be the first centroids
    # centroids = np.array([[0.76472332, 0.73025491],[1.54885617, 1.86759422]])
    centroids =  dataset[np.random.randint(0, num_instances - 1, k)] # TODO: may pick 2 same centroids
    # TODO: check if I should omit the '-1' to include the last element since...
    # randint()fills with random integers from low (inclusive) to high (*exclusive*)
    #
    # set these to a list of past centroids (to show progress over time)
    history_centroids.append(centroids)


    # store clusters
    belongs_to = np.zeros((num_instances, 1))


    epochs = 5
    for i in range(epochs): #TODO: Other ways to stop? Siraj's code uses while norm > epsilon
        # for every point
        for index_instance, instance in enumerate(dataset):
            # define a distance vector of size k
            dis_vec = np.zeros((k,1))
            # for each centroid
            for index_centroid, centroid in enumerate(centroids): # index_centroids numbers off all the centroids (should be same for the whole code)
                # compute distance between instance and centroid
                dis_vec[index_centroid] = euclidean(centroid, instance) # TODO: seems to mess up for some runs
                # dis_vec[index_centroid] = euclidean(centroid, np.array([1.07324343, 1.23179161]))
                '''
                critical observation: there are [0.     ] in my dis_vec where there shouldn't be (already did my checks)
                
                print(centroid)
                print(instance)
                print(dis_vec)
                '''
            # find the smallest distance to a centroid, and assign the instance to that centroid
            belongs_to[index_instance, 0] = np.argmin(dis_vec) # argmin return index of smallest element
            # NOTE: index of dis_vec is index_centroid
        plot(belongs_to, centroids)
        # for each cluster (k of them)
        for index in range(len(centroids)):
            # get all the points assigned to a cluster
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index] # returns indices of elements in dataset array that belong to the cluster
            # find the mean of those points, this is our new centroid
            centroid = np.mean(dataset[instances_close],axis=0)
            centroids[index, :] = centroid
            #print("centroids {} {}".format(index, centroid))

        #test
        '''
        for i in range(len(dataset)):
            print("The point ", dataset[i])
            print("belongs to ", belongs_to[i])
        
        '''


    return centroids, belongs_to


# plots the colored points based on assigned cluster
def plot(belongs_to, centroids):
    # we have two colours for each cluster
    colors = ['r', 'g']
    # for each cluster
    for cluster_index in range(2):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == cluster_index] # array of indices to dataset
        # print(instances_close)
        for i in range(len(instances_close)):
            plt.plot(dataset[instances_close[i]][0], dataset[instances_close[i]][1], colors[cluster_index] + 'o')

            # plt.plot(dataset[instance_index], colors[cluster_index] + 'o') # o is for circle markers
    # plt.plot(dataset[0]) f you provide a single list or array to the plot() command,
    # matplotlib assumes it is a sequence of y values, and automatically generates the x values for you.
    for i in range(2):
        plt.plot(centroids[i][0], centroids[i][1], 'bo')
    plt.show()







dataset = np.loadtxt("durudataset.txt")
# test
# np.random.shuffle(dataset)


centroids , belongs_to = kmeans(2, dataset)
#plot(belongs_to, centroids)
#print(belongs_to)
'''
#plt.show() opens a new figure ie window
a = np.array([1.07324343, 1.23179161])
a_0 = np.array([0.76472332, 0.73025491])
a_1 = np.array([1.54885617, 1.86759422])
print(np.linalg.norm(a-a_0)) # 0.5888325056595484
print(np.linalg.norm(a-a_1)) # 0.7940103508979716
# could the centroid indexes have been swapped????
# so should be classified to [1] but it was instead [0] hmmmmm
'''


'''
for i in range(len(dataset)):
    print("The point ", dataset[i])
    print("belongs to ", belongs_to[i])
# Observations:
# points (x,y) with x>1 and y>1 got classfied to cluster '0' and
# those with x<1 and y<1 got classfied to cluster '1'
# seems to be working
'''

'''
what i think is wrong:

[1.07324343 1.23179161] instance
[[0.58883251] dis from centroid 0
 [0.79401035]] dis from centroid 1
 
 what i think is correct:
 
 [1.35650893 1.47948455] instance
[[0.95475403] dis from centroid 0
 [0.43315884]] dis from centroid 1
 
 both should by right from visual inspection go to centroid 1
 
'''