import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.metrics import classification_report

class CARTNode:
    def __init__(self, split_attr=-1, depth=0, label=None, parent=None, continous=False):
        self.split_attr = split_attr
        self.split_threshold = None
        self.label = label
        self.parent = parent
        self.samples: list = []
        self.predicts: list = []
        self.depth = depth
        self.continous = False
        self.children = {}

class CART:
    def __init__(self, min_sample_count=2, max_depth=np.inf):
        self.root: CARTNode = None
        self.leaves: list = []
        self.X = None
        self.y = None
        self.n_features = None
        self.continous_attr: list = []
        self.min_sample_count = min_sample_count
        self.max_depth = max_depth
        self.n_leaves = 0
        self.depth = 0

    def fit(self,X,y):
        self.X, self.y = np.array(X), np.array(y).reshape(-1)
        self.n_features = self.X.shape[1]
        self.continous_attr = [CART.is_number(x) for x in self.X[0]]
        self.root = CARTNode()
        self.root.samples = np.arange(X.shape[0])
        stack = [(self.root, list(range(self.n_features)))]

        while len(stack)>0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.samples], self.y[node.samples]
            unique_labels, label_counts = np.unique(node_y, return_counts=True)
            prior_label = unique_labels[np.argmax(label_counts)]
            if len(unique_labels) == 1 \
                    or len(attr_set) == 0 \
                    or len(np.unique(node_X[:, attr_set], axis=0)) == 1 \
                    or len(node.samples) <= self.min_sample_count \
                    or node.depth >= self.max_depth:
                node.label = prior_label
                self.leaves.append(node)
            else:
                node.split_attr, (_, node.split_threshold) = self.__get_split_attr(node.samples, attr_set)
                attr_set.remove(node.split_attr)
                if self.continous_attr[node.split_attr]:
                    node.continous = True
                    attr_vector = node_X[:, node.split_attr].astype(float)
                    idx_left = attr_vector<=node.split_threshold
                else:
                    idx_left = node_X[:, node.split_attr]==node.split_threshold
                idx_right = np.logical_not(idx_left)
                left_samples, right_samples = node.samples[idx_left], node.samples[idx_right]
                if min(len(left_samples), len(right_samples)) < self.min_sample_count:
                    node.label = prior_label
                    self.leaves.append(node)
                    continue
                left_node = CARTNode(depth=node.depth+1, parent=node)
                right_node = CARTNode(depth=node.depth+1, parent=node)
                left_node.samples = left_samples
                right_node.samples = right_samples
                node.children.update({True:left_node, False:right_node})
                stack.append((node.children[False], attr_set.copy()))
                stack.append((node.children[True], attr_set.copy()))
        self.n_leaves = len(self.leaves)
        self.depth = max([leaf.depth for leaf in self.leaves])

    def predict(self,X):
        X = np.array(X).reshape(-1, self.n_features)
        for leaf in self.leaves:
            leaf.predicts = None
        self.root.predicts = np.arange(X.shape[0])
        stack = [self.root]

        while len(stack) > 0:
            node: CARTNode = stack.pop()
            if node.label is None:
                data = X[node.predicts]
                if node.continous:
                    left_idx = data[:, node.split_attr].astype(float) <= node.split_threshold
                else:
                    left_idx = data[:, node.split_attr] == node.split_threshold
                right_idx = np.logical_not(left_idx)
                node.children[True].predicts = node.predicts[left_idx]
                node.children[False].predicts = node.predicts[right_idx]
                stack.append(node.children[False])
                stack.append(node.children[True])
        pred = np.zeros(X.shape[0])
        for leaf in self.leaves:
            if leaf.predicts is not None:
                pred[leaf.predicts] = leaf.label
        return pred

    def ccp_pruning(self):
        pass

    def __get_split_attr(self, sample_idx, attr_set):
        X,y = self.X[sample_idx], self.y[sample_idx]
        gini_dict = {}
        for attr in attr_set:
            attr_vector = X[:, attr]
            if self.continous_attr[attr]:
                attr_vector = attr_vector.astype(float)
                split_value_set = np.unique(attr_vector)
                if len(split_value_set) == 1:
                    gini_dict[attr] = (split_value_set[0], -np.inf)
                    continue
                split_value_set = [(split_value_set[i]+split_value_set[i+1])/2 for i in range(0, len(split_value_set)-1)]
                gini_values = [
                    -len(y_left:=y[attr_vector<=v]) * CART.__cal_gini(y_left) - len(y_right:=y[attr_vector>v]) * CART.__cal_gini(y_right)
                    for v in split_value_set
                ]
                max_idx = np.argmax(gini_values)
                split_threshold = split_value_set[max_idx]
                gini_dict[attr] = (gini_values[max_idx], split_threshold)

            else:
                unique_values = np.unique(attr_vector)
                gini_values = [
                    -len(y1:=y[attr_vector==attr_v]) * CART.__cal_gini(y1) - len(y2:=y[attr_vector!=attr_v]) * CART.__cal_gini(y2)
                    for attr_v in unique_values
                ]
                max_idx = np.argmax(gini_values)
                split_threshold = gini_values[max_idx]
                gini_dict[attr] = (gini_values[max_idx], split_threshold)
        return max(gini_dict.items(), key=lambda item:item[1][0])


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
    def __cal_gini(labels):
        label_counts = np.unique(labels, return_counts=True)[1].astype(float)
        percents = label_counts/np.sum(label_counts)
        return 1.0 - np.sum(percents**2)

if __name__ == '__main__':
    name = 'iris'
    dataset = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits,
    }[name]()
    X,y = dataset.data, dataset.target
    train_X, test_X, train_y, test_y = train_test_split(X,y,train_size=0.7,random_state=2222)
    model = CART()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print(classification_report(test_y, pred, target_names=dataset.target_names))


