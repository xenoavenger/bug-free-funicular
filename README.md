# graph-experiments
Some experiments with machine learning on graphs.  Currently, this repository only contains a script for evaluating Logistic Regression and Random Forest classifiers on discriminating between graphs trained with the Erdos-Renyi and planted-partition random graph models using four different ways of generating features from the distribution of all-pairs shortest-path lengths for each graph.
You can run the script like so:

```
$ evaluate_shortest_path_features.py --n-graphs 100 --n-nodes 100 --p-edge 0.2
```
