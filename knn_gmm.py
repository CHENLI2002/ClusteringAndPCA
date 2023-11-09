# Generate all 5 data sets,
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

sigma = [0.5, 1, 2, 4, 8]
mean_0, covariance_0 = [-1, -1], np.array([[2, 0.5], [0.5, 1]])
mean_1, covariance_1 = [1, -1], np.array([[1, -0.5], [-0.5, 2]])
mean_2, covariance_2 = [0, 1], np.array([[1, 0], [0, 2]])

dataset = []

for s in sigma:
    data_0 = np.random.multivariate_normal(mean=mean_0, cov=s * covariance_0, size=100)
    label_0 = np.zeros((100, 1))
    complete_data_0 = np.hstack((data_0, label_0))
    data_1 = np.random.multivariate_normal(mean=mean_1, cov=s * covariance_1, size=100)
    label_1 = np.ones((100, 1))
    complete_data_1 = np.hstack((data_1, label_1))
    data_2 = np.random.multivariate_normal(mean=mean_2, cov=s * covariance_2, size=100)
    label_2 = np.ones((100, 1)) * 2
    complete_data_2 = np.hstack((data_2, label_2))
    all = np.vstack((complete_data_0, complete_data_1, complete_data_2))
    dataset.append(all)

# Define K clustering
def k_means(data, c0, c1, c2):
    if c0 is None:
        centroids = data[np.random.choice(300, 3, replace=False), :-1]
        centroid_0 = centroids[0]
        centroid_1 = centroids[1]
        centroid_2 = centroids[2]
    else:
        centroid_0 = c0
        centroid_1 = c1
        centroid_2 = c2

    cluster0 = []
    cluster1 = []
    cluster2 = []


    for point in data:
        dis_0 = np.linalg.norm(point[:2] - centroid_0)
        dis_1 = np.linalg.norm(point[:2] - centroid_1)
        dis_2 = np.linalg.norm(point[:2] - centroid_2)

        closet = np.argmin([dis_0, dis_1, dis_2])

        if closet == 0:
            cluster0.append(point)
        elif closet == 1:
            cluster1.append(point)
        else:
            cluster2.append(point)

    new_c_0 = np.mean(cluster0,axis=0)[:2]
    new_c_1 = np.mean(cluster1,axis=0)[:2]
    new_c_2 = np.mean(cluster2,axis=0)[:2]

    if abs(np.linalg.norm(new_c_0 - centroid_0)) > 0.001 \
            and abs(np.linalg.norm(new_c_1 - centroid_1)) >  0.001 \
            and abs(np.linalg.norm(new_c_2 - centroid_2)) > 0.001:
        return k_means(data, new_c_0, new_c_1, new_c_2)
    else:
        sum = 0
        accuracy = 0

        for point in cluster0:
            dis_0 = np.linalg.norm(point[:2] - new_c_0)
            sum += dis_0
            if point[2] != 0:
                accuracy += 1
        for point in cluster1:
            dis_1 = np.linalg.norm(point[:2] - new_c_1)
            sum += dis_1
            if point[2] != 1:
                accuracy += 1
        for point in cluster2:
            dis_2 = np.linalg.norm(point[:2] - new_c_2)
            sum += dis_2
            if point[2] != 2:
                accuracy += 1

        return sum, accuracy / (np.array(data).shape[0])



# Define GMM
def gmm(data):
    # Initialize
    indices = np.random.choice(np.array(data).shape[0], 3, replace=False)
    means = data[indices, :-1]
    cov = [[np.random.uniform(1, 0), np.random.uniform(1, 0)],
           [np.random.uniform(1, 0), np.random.uniform(1, 0)]] * 3
    phi = [1/3, 1/3, 1/3]


    num_points = len(data)
    w0 = []
    w1 = []
    w2 = []
    prev_log = math.inf
    log_likelihood = 0
    # e step
    while abs(prev_log - log_likelihood) > 0.001:
        print(log_likelihood)
        prev_log = log_likelihood
        mvn_0 = multivariate_normal(mean=means[0], cov=cov[0])
        mvn_1 = multivariate_normal(mean=means[1], cov=cov[1])
        mvn_2 = multivariate_normal(mean=means[2], cov=cov[2])
        w0 = []
        w1 = []
        w2 = []
        for point in data:
            a_p = point[:-1]
            w = [mvn_0.pdf(a_p)*phi[0], mvn_1.pdf(a_p)*phi[1], mvn_2.pdf(a_p)*phi[2]]
            sum_w = np.sum(w)
            w0.append(w[0] / sum_w)
            w1.append(w[1]/ sum_w)
            w2.append(w[2] / sum_w)
        sums = np.array([np.sum(np.array(w0)), np.sum(np.array(w1)), np.sum(np.array(w2))])
        w0 = np.array(w0)
        w1 = np.array(w1)
        w2 = np.array(w2)
        phi = sums / num_points
        means = [np.sum(w0[:, np.newaxis] * data[:, :-1], axis=0) / np.sum(w0),
                 np.sum(w1[:, np.newaxis] * data[:, :-1], axis=0) / np.sum(w1),
                 np.sum(w2[:, np.newaxis] * data[:, :-1], axis=0) / np.sum(w2)]
        for i in range(3):
            diff = data[:, :-1] - means[i]
            if i == 0:
                cov[i] = np.dot((w0[:, np.newaxis] * diff).T, diff) / np.sum(w0)
            elif i == 1:
                cov[i] = np.dot((w1[:, np.newaxis] * diff).T, diff) / np.sum(w1)
            else:
                cov[i] = np.dot((w2[:, np.newaxis] * diff).T, diff) / np.sum(w2)
        log_likelihood = np.sum(np.log(np.array(w0) + np.array(w1) + np.array(w2)))

    log_likelihood = np.sum(np.log(np.array(w0) + np.array(w1) + np.array(w2)))
    predicted_labels = np.argmax([w0, w1, w2], axis=0)
    true_labels = data[:, -1]
    accuracy = np.sum(predicted_labels == true_labels) / num_points

    return log_likelihood, accuracy

# Write evalutaion code　
knn_result_obj = []
knn_result_acc = []
for data in dataset:
    obj, acc = k_means(data, None, None, None)
    knn_result_obj.append(obj)
    knn_result_acc.append(acc)


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(sigma, knn_result_obj, '-o', c='r')
plt.xlabel('sigma')
plt.ylabel('knn-obj')
plt.title("knn-result")
plt.savefig("knn_gmm_result/knn_obj.png")

plt.subplot(1, 2, 2)
plt.plot(sigma, knn_result_acc,'-o', c='r')
plt.xlabel('sigma')
plt.ylabel('knn-accuracy')
plt.title("knn-result_accuracy")
plt.savefig("knn_gmm_result/knn_acc.png")
plt.show()

gmm_obj = []
gmm_acc = []
for data in dataset:
    obj, acc = gmm(data)
    gmm_obj.append(obj)
    gmm_acc.append(acc)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(sigma, gmm_obj, '-o', c='r')
plt.xlabel('sigma')
plt.ylabel('gmm_obj')
plt.title("gmm-result")
plt.savefig("knn_gmm_result/gmm_obj.png")

plt.subplot(1, 2, 2)
plt.plot(sigma, gmm_acc,'-o', c='r')
plt.xlabel('sigma')
plt.ylabel('gmm_acc')
plt.title("gmm-result_accuracy")
plt.savefig("knn_gmm_result/gmm_acc.png")
plt.show()
