import matplotlib as mpl
import matplotlib.ticker as ticker

mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

from numpy import *
import numpy.random
from numpy import linalg as LA
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos) * 2 - 1

# validation_data_unscaled = data[train_idx[6000:], :].astype(float)
# validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
# validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

eight_train_data = train_data[train_labels == 1]
zero_train_data = train_data[train_labels == -1]


def PCA(x):
    x = sklearn.preprocessing.scale(x, axis=0, with_std=False)
    x_t = numpy.transpose(x)
    x_t_dot_x = numpy.dot(x_t, x)
    w, v = LA.eigh(x_t_dot_x)
    return w, numpy.transpose(v)


def plot_mean_image_of_data(x, label):
    x = numpy.transpose(x)
    x = x.mean(axis=1)
    plt.imshow(reshape(x, (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '{0}_mean_data.png'.format(label)))


def print_first_5_eigenvectors(x, label):
    _, v = PCA(x)
    for i in range(5):
        plt.imshow(reshape(v[-1 - i], (28, 28)), interpolation="nearest")
        plt.savefig(os.path.join(dir_path, '{0}_{1}_eigenvectors.png'.format(label, i + 1)))


def plot_first_100_eigenvalues(x, label):
    w, _ = PCA(x)
    w = w[::-1]
    w = w[:100]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(numpy.arange(1, 101), w, 'r-', label='first 100 eigenvalues of {0}'.format(label))
    plt.legend()
    plt.ylabel('eigenvalues', fontsize=16)
    fig.savefig(os.path.join(dir_path, '{0}_first_100_eigenvalues.png'.format(label)))
    fig.clf()


def analyze_data(data, label):
    plot_mean_image_of_data(data, label)
    print_first_5_eigenvectors(data, label)
    plot_first_100_eigenvalues(data, label)


def assignment_1a():
    analyze_data(eight_train_data, "eight")


def assignment_1b():
    analyze_data(zero_train_data, "zero")


def assignment_1c():
    analyze_data(train_data, "all")


def assignment_1d():
    _, v = PCA(train_data)
    v1 = reshape(v[-1], (784, 1))
    v2 = reshape(v[-2], (784, 1))
    eights_v1 = numpy.dot(eight_train_data, v1)
    eights_v2 = numpy.dot(eight_train_data, v2)
    zero_v1 = numpy.dot(zero_train_data, v1)
    zero_v2 = numpy.dot(zero_train_data, v2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(eights_v1, eights_v2, label='eight', facecolor='red', s=4)
    ax.scatter(zero_v1, zero_v2, label='zero', facecolor='blue', s=4)
    plt.legend()
    plt.title('scatter of first 2 eigenvectors', fontsize=16)
    fig.savefig(os.path.join(dir_path, 'scatter.png'))
    fig.clf()


def plt_images(origin, K, label):
    fig = plt.figure()
    a = fig.add_subplot(1, 4, 1)
    plt.imshow(reshape(origin, (28, 28)))
    a.set_title('Origin')
    a = fig.add_subplot(1, 4, 2)
    plt.imshow(reshape(K[0], (28, 28)))
    a.set_title('K=10')
    a = fig.add_subplot(1, 4, 3)
    plt.imshow(reshape(K[1], (28, 28)))
    a.set_title('K=20')
    a = fig.add_subplot(1, 4, 4)
    plt.imshow(reshape(K[2], (28, 28)))
    a.set_title('K=30')
    fig.savefig(os.path.join(dir_path, '{0}_decomposition.png'.format(label)))
    fig.clf()


def project_on_eigenvectors(x, v, k):
    new_x = numpy.zeros((1, 784))
    for i in range(k):
        add_to_x = reshape(numpy.dot(x, v[-1 - i]) * v[-1 - i], (1, 784))
        new_x += add_to_x
    return new_x


def reconstruct_2_from_data(data, v, label):
    for i in range(2):
        origin = data[i]
        k_reconstruction = []
        for j in range(3):
            k_reconstruction.append(project_on_eigenvectors(origin, v, (j + 1) * 10))
        plt_images(origin, k_reconstruction, "{0}_{1}".format(label, i + 1))


def assignment_1e():
    _, v = PCA(train_data)
    reconstruct_2_from_data(eight_train_data, v, "eight")
    reconstruct_2_from_data(zero_train_data, v, "zero")


def vectors_total_distance(x):
    distance = 0
    for vector in x:
        distance += numpy.linalg.norm(vector)
    return distance


def assignment_1f():
    _, v = PCA(train_data)
    current_distances = train_data
    distances = [vectors_total_distance(current_distances)]
    eingenvectors = v[::-1]
    for i in range(len(eingenvectors)):
        en = reshape(eingenvectors[i], (784, 1))
        dot_prdc = numpy.dot(train_data, en)
        current_distances -= numpy.dot(dot_prdc, reshape(en, (1, 784)))
        distances.append(vectors_total_distance(current_distances))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000000))
    ax.yaxis.set_major_formatter(ticks)
    ax.set_title('PCA objective', fontsize=16)
    ax.plot(numpy.arange(len(eingenvectors) + 1), distances, 'r-')
    plt.xlabel('number of eigenvalue used', fontsize=16)
    plt.ylabel('distance $10^6$', fontsize=16)
    fig.savefig(os.path.join(dir_path, 'PCA_objective.png'))
    fig.clf()


assignment_1a()
assignment_1b()
assignment_1c()
assignment_1d()
assignment_1e()
assignment_1f()

if len(sys.argv) < 2:
    print("Please enter which part do you want to execute - a, b, c, d or all")
    exit()

cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['a', 'b', 'c', 'd', 'e', 'all']:
        print("Unknown argument %s. please run with a, b, c, d or all" % cmd)
        exit()

if 'a' in cmds or 'all' in cmds:
    assignment_1a()
if 'b' in cmds or 'all' in cmds:
    assignment_1b()
if 'c' in cmds or 'all' in cmds:
    assignment_1c()
if 'd' in cmds or 'all' in cmds:
    assignment_1d()
if 'e' in cmds or 'all' in cmds:
    assignment_1e()
if 'f' in cmds or 'all' in cmds:
    assignment_1f()
