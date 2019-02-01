import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import networkx as nx

def plot_graphs(flname, graphs):
    plt.clf()
    G = nx.disjoint_union_all(graphs)
    c = [random.random() for i in xrange(nx.number_of_nodes(G))]
    nx.draw(G,
            pos=nx.nx_pydot.pydot_layout(G, prog="neato"),
            node_size=50,
            node_color=c,
            vmin=0.0,
            vmax=1.0,
            cmap=plt.get_cmap("Vega20c"))
    plt.savefig(flname, DPI=200)

def plot_roc(flname, curves, title):
    plt.clf()
    for label, fpr, tpr in curves:
        plt.plot(fpr, tpr, label=label)
    plt.legend(loc="lower right")
    plt.title(title, fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.savefig(flname, DPI=200)

def plot_histogram_average_sp_lengths(flname, graph_sets):
    lengths = []
    plt.clf()
    for label, graphs in graph_sets:
        for g in graphs:
            avg_l = nx.average_shortest_path_length(g)
            lengths.append(avg_l)
        sns.distplot(lengths, label=label)
    plt.xlabel("Average Shortest-Paths Length", fontsize=16)
    plt.ylabel("Count (graph)", fontsize=16)
    plt.legend(loc="upper right")
    plt.savefig(flname, DPI=200)

def plot_sp_dist_distances(flname, sp_distr_sets):
    between_distances = []
    all_within_distances = dict()
    for i, (label_1, distr_set_1) in enumerate(sp_distr_sets):
        within_distances = []
        print "Computing within distances for", label_1
        for j, distr_1 in enumerate(distr_set_1[:-1]):
            for distr_2 in distr_set_1[j+1:]:
                diff = distr_1 - distr_2
                d = np.dot(diff, diff)
                within_distances.append(d)
        all_within_distances[label_1] = within_distances

        for label_2, distr_set_2 in sp_distr_sets[i+1:]:
            print "Computing between distances for", label_1, "and", label_2
            for distr_1 in distr_set_1:
                for distr_2 in distr_set_2:
                    diff = distr_1 - distr_2
                    d = np.dot(diff, diff)
                    between_distances.append(d)

    plt.clf()
    sns.distplot(between_distances, hist=True, kde=True, norm_hist=True, label="between")
    for label, distances in all_within_distances.iteritems():
        sns.distplot(distances, hist=True, kde=True, norm_hist=True, label=label)

    plt.xlabel("Average Shortest-Paths Length", fontsize=16)
    plt.ylabel("Count (graph)", fontsize=16)
    plt.legend(loc="upper right")
    plt.savefig(flname, DPI=200)
