import numpy as np
from functools import reduce

class C45Node:
    def __init__(self, split_attr=-1, label=None, parent=None, depth=0, continous=False):
        self.samples: list = None
        self.predicts: list = None
        self.children: dict = {}
        self.parent = parent
        self.label = label
        self.split_attr = split_attr

        # for continous values
        self.continous: bool = continous
        self.threshold: float = None

        # for pre-pruning
        self.depth = depth
        self.n_leaves = 0

class C45TreeClassifier:
    def __init__(self):
        super(C45TreeClassifier, self).__init__()
        self.X = None
        self.y = None
        self.n_features = None
        self.root: C45Node = None
        self.leaves: list = []
        self.n_leaves = 0
        self.attr_continous = None
        self.depth = 0
        self.max_depth = np.inf
        self.min_samples_split = 2

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y).reshape(-1)
        self.attr_continous = [C45TreeClassifier.is_number(x) for x in X[0]]
        self.n_features = self.X.shape[1]
        self.root: C45Node = C45Node()
        self.root.samples = np.arange(len(X))

        stack = [(self.root, list(range(self.n_features)))]
        while len(stack) > 0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.samples], self.y[node.samples]
            unique_labels, label_counts = np.unique(node_y, return_counts=True)
            prior_label = unique_labels[np.argmax(label_counts)]
            if len(unique_labels)==1 \
                    or len(np.unique(node_X[:, attr_set],axis=0))==1 \
                    or len(attr_set)==0\
                    or len(node.samples)<=self.min_samples_split\
                    or node.depth>=self.max_depth:
                node.label = prior_label
                self.leaves.append(node)
            else:
                node.split_attr = self.__get_split_attr(node.samples, attr_set)
                attr_set.remove(node.split_attr)

                # for continous attr, get node threshold and split sample to left & right according to the values
                if self.attr_continous[node.split_attr]:
                    node.continous = True
                    all_values = node_X[:, node.split_attr].astype(float)
                    idx_left = all_values<= node.threshold
                    idx_right = np.logical_not(idx_left)
                    left_samples = node.samples[idx_left]
                    right_samples = node.samples[idx_right]

                    if min(len(left_samples), len(right_samples))<self.min_samples_split:
                        node.label = prior_label
                        self.leaves.append(node)
                        continue

                    left_node = C45Node(parent=node, depth=node.depth+1)
                    right_node = C45Node(parent=node, depth=node.depth+1)
                    left_node.samples, right_node.samples = left_samples, right_samples
                    node.children.update({True:left_node, False:right_node})
                    stack.append((right_node, attr_set.copy()))
                    stack.append((left_node, attr_set.copy()))

                else:
                    # split & get |V| children nodes, so use self.X(for all possible attr_vals) instead of node_X
                    unique_attrs = np.unique(self.X[:, node.split_attr])
                    for attr in unique_attrs:
                        childNode = C45Node(depth=node.depth+1, parent=node)
                        childNode.samples = node.samples[X[:, node.split_attr]==attr]
                        if len(childNode.samples) == 0:
                            childNode.label = prior_label
                            self.leaves.append(childNode)
                        else:
                            stack.append((childNode, attr_set.copy()))
                        node.children[attr] = childNode
        self.n_leaves = len(self.leaves)
        self.depth = max([leaf.depth for leaf in self.leaves])
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        self.root.predicts = np.arange(len(X))
        for leaf in self.leaves:
            leaf.predicts = None
        stack = [self.root]

        while len(stack) > 0:
            node = stack.pop(0)
            if node.label is None:
                data = X[node.predicts]
                # for continous values
                if node.continous:
                    idx_left = data[:,node.split_attr].astype(float) <= node.threshold
                    idx_right = np.logical_not(idx_left)
                    node.children[True].predicts = idx_left
                    node.children[False].predicts = idx_right
                    stack.append(node.children[True])
                    stack.append(node.children[False])
                else:
                    for attr, childNode in node.children.items():
                        childNode.predicts = node.predicts[data[:,node.split_attr]==attr]
                        stack.append(childNode)
        pred = np.zeros(len(X))
        for leaf in self.leaves:
            if leaf.predicts is not None:
                pred[leaf.predicts] = leaf.label
        return pred

    def pep_pruning(self):
        pass

    def __get_split_attr(self, samples, attr_set):
        X,y = self.X[samples], self.y[samples]
        ent0 = C45TreeClassifier.cal_entropy(y)
        attr_gain_dict: dict = {}
        for attr in attr_set:
            if self.attr_continous[attr]:
                vals = X[:,attr].astype(float)
                vals_set = np.unique(vals)
                if len(vals_set)==1:
                    continue
                thresholds = [(vals_set[i]+vals_set[i+1])/2 for i in range(len(vals_set)-1)]
                entropys = [
                    -len(y_left:=y[vals<=t]) * C45TreeClassifier.cal_entropy(y_left)
                    -len(y_right:=y[vals>t]) * C45TreeClassifier.cal_entropy(y_right)
                    for t in thresholds
                ]
                split_idx = np.argmax(entropys)
                split_threshold = thresholds[split_idx]
                gain = ent0 + entropys[split_idx]/len(samples)
                # gain-ratio: gain/(entropy of True/False)
                attr_gain_dict[attr] = (
                    gain, gain/C45TreeClassifier.cal_entropy(vals<=split_threshold), split_threshold
                )
            else:
                unique_attr, attr_counts = np.unique(X[:,attr])
                gain = ent0 - sum(
                    attr_counts[i] * C45TreeClassifier.cal_entropy(y[X[:,attr]==unique_attr[i]]) for i in range(len(unique_attr))
                )/len(X)
                attr_gain_dict[attr] = (
                    gain, gain/C45TreeClassifier.cal_entropy(X[:,attr]), None
                )

        # get average gain
        avg = np.mean([item[0] for item in attr_gain_dict.values()])
        # sort with gain-ratio
        returnItem = None
        # search from biggest gain-ratio, return the first one whose gain > average-gain
        for item in sorted(attr_gain_dict.items(), key=lambda item:item[1][1], reverse=True):  # big -> small
            if item[1][0] > avg:
                returnItem = item
                break
        return returnItem

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    @staticmethod
    def cal_entropy(subset_labels):
        label_counts = np.unique(subset_labels, return_counts=True)[1].astype(float)
        percent = label_counts/np.sum(label_counts)
        return -np.sum(percent * np.log2(percent))