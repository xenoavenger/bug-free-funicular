import argparse
import os
import time

import networkx as nx
import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from ml import *
from plots import *

def generate_er_graphs(n, v, p):
    graphs = []
    for i in xrange(n):
        G = nx.fast_gnp_random_graph(v,
                                     p)
        graphs.append(G)
    
    return graphs

def generate_pp_graphs(n, v, p_1, multiplier):
    p_2 = multiplier * p_1
    q_2 = 2. * p_1 - p_2 - 2. * (p_1 - p_2) / n
    graphs = []
    for i in xrange(args.n_graphs):
        G = nx.planted_partition_graph(2,
                                       n / 2,
                                       p_2,
                                       q_2)
        graphs.append(G)

    return graphs

def compute_sp_distr(g):
    sps = nx.shortest_path_length(g)
    flattened = []
    for s in sps.iterkeys():
        for t, sp_len in sps[s].iteritems():
            flattened.append(sp_len)
    distr = np.bincount(flattened)
    distr = distr / LA.norm(distr)

    return distr


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-graphs",
                        type=int,
                        required=True)

    parser.add_argument("--n-nodes",
                        type=int,
                        required=True)

    parser.add_argument("--p-edge",
                        type=float,
                        required=True)

    parser.add_argument("--pp-multiplier",
                        type=float,
                        default=1.6,
                        required=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists("figures"):
        os.mkdir("figures")
    
    start_time = time.clock()
    er_graphs = generate_er_graphs(args.n_graphs,
                                   args.n_nodes,
                                   args.p_edge)
    
    pp_graphs = generate_pp_graphs(args.n_graphs,
                                   args.n_nodes,
                                   args.p_edge,
                                   args.pp_multiplier)

    print "Computing shortest path distributions"
    er_sp_distributions = list(map(compute_sp_distr, er_graphs))
    pp_sp_distributions = list(map(compute_sp_distr, pp_graphs))

    plot_graphs("figures/graphs.png",
                [er_graphs[0],
                 pp_graphs[0]])

    plot_histogram_average_sp_lengths("figures/avg_sp_hist.png",
                                      [("ER", er_graphs),
                                       ("PP", pp_graphs)])

    plot_sp_dist_distances("figures/sp_distr_dist_hist.png",
                           [("ER", er_sp_distributions),
                            ("PP", pp_sp_distributions)])

    print "Machine learning"
    print

    sgd = SGDClassifier(loss="log",
                        penalty="l2",
                        n_iter=1000)
    rf = RandomForestClassifier(n_estimators=100)
    X, y = create_features_1(er_sp_distributions,
                             pp_sp_distributions)
    
    print "Testing feature type 1 with Logistic Regression"
    lr_fpr_1, lr_tpr_1 = evaluate_classifier(sgd, X, y)
    print "Testing feature type 1 with Random Forest"
    rf_fpr_1, rf_tpr_1 = evaluate_classifier(rf, X, y)

    X, y = create_features_2(er_sp_distributions,
                             pp_sp_distributions)
    print "Testing feature type 2 with Logistic Regression"    
    lr_fpr_2, lr_tpr_2 = evaluate_classifier(sgd, X, y)
    print "Testing feature type 2 with Random Forest"
    rf_fpr_2, rf_tpr_2 = evaluate_classifier(rf, X, y)

    X, y = create_features_3(er_sp_distributions,
                             pp_sp_distributions)
    print "Testing feature type 3 with Logistic Regression"
    lr_fpr_3, lr_tpr_3 = evaluate_classifier(sgd, X, y)
    print "Testing feature type 3 with Random Forest"
    rf_fpr_3, rf_tpr_3 = evaluate_classifier(rf, X, y)


    X, y = create_features_4(er_sp_distributions,
                             pp_sp_distributions)
    print "Testing feature type 4 with Logistic Regression"
    lr_fpr_4, lr_tpr_4 = evaluate_classifier(sgd, X, y)
    print "Testing feature type 4 with Random Forest"
    rf_fpr_4, rf_tpr_4 = evaluate_classifier(rf, X, y)

    plot_roc("figures/lr_roc.png",
             [("1", lr_fpr_1, lr_tpr_1),
              ("2", lr_fpr_2, lr_tpr_2),
              ("3", lr_fpr_3, lr_tpr_3),
              ("4", lr_fpr_4, lr_tpr_4)],
             "Logistic Regression")

    plot_roc("figures/rf_roc.png",
             [("1", rf_fpr_1, rf_tpr_1),
              ("2", rf_fpr_2, rf_tpr_2),
              ("3", rf_fpr_3, rf_tpr_3),
              ("4", rf_fpr_4, rf_tpr_4)],
             "Random Forest")
    
    end_time = time.clock()

    print "Elapsed time:", (end_time - start_time)
