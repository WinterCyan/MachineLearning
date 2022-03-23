import numpy as np
from functools import reduce

class ID3Node:
    def __init__(self, split_attr=-1, parent=None):
        self.split_attr: int = split_attr
        self.samples: list = None
        self.predicts: list = None
        self.parent = None
        self.children: dict = {}
        self.label_val = None


class ID3TreeClassifier:
    def __init__(self):
        super().__init__()
        self.n_features = None
        self.X = None
        self.y = None
        self.root = None
        self.leaves: list = []
        self.n_leaves = 0

    def fit(self, X, y):
        """
        :param X: training samples
        :param y: training labels
        """
        self.X, self.y = np.array(X), np.array(y).reshape(-1)
        self.n_features = self.X.shape[1]
        self.root = ID3Node()
        self.root.samples = np.arange(len(self.X))  # define samples as list of idx

        # stack nodes are pairs of (node, attr_set)
        stack = [(self.root, list(range(self.n_features)))]
        while len(stack)>0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.samples], self.y[node.samples]
            unique_labels, label_counts = np.unique(node_y, return_counts=True)
            prior_label = unique_labels[np.argmax(label_counts)]
            # all sample has same labels, or has no attr to split, or has same attributes combines
            if len(unique_labels)==1 or len(attr_set)==0 or len(np.unique(node_X[:,attr_set], axis=0))==1:
                node.label_val = prior_label
                self.leaves.append(node)
            else:
                # get split attribute
                node.split_attr = self.__get_split_attr(node.samples, attr_set)
                # COPY is neccessary
                copy_set = attr_set.copy()
                copy_set.remove(node.split_attr)
                unique_attr_vals = np.unique(self.X[:,node.split_attr])
                # for every possible values of split attribute
                for attr in unique_attr_vals:
                    # create child node
                    node.children[attr] = ID3Node(parent=node)
                    # assign samples
                    children_samples = node.samples[node_X[:,node.split_attr]==attr]
                    node.children[attr].samples = children_samples
                    # check new node is a leaf or not
                    if len(children_samples)==0:
                        node.children[attr].label_val = prior_label
                        self.leaves.append(node.children[attr])
                    else:
                        stack.append((node.children[attr], copy_set))
        self.n_leaves = len(self.leaves)
        return self

    def predict(self, X):
        """
        :param X: samples to predict label
        :return: labels predicted
        """
        X = np.array(X).reshape(-1, self.n_features)
        for leaf in self.leaves:
            leaf.predicts = None
        self.root.predicts = np.arange(len(X))
        stack = [self.root]

        while len(stack) > 0:
            node = stack.pop()
            if node.label_val is None:
                data = X[node.predicts]
                for attr_val, child in node.children.items():
                    child.predicts = node.predicts[data[:,node.split_attr]==attr_val]
                    stack.append(child)
        pred = np.zeros(len(X))
        for leaf in self.leaves:
            if leaf.predicts is not None:
                pred[leaf.predicts] = leaf.label_val
        return pred

    def rep_pruning(self, valid_X, valid_y):
        """
        :param valid_X: validation samples
        :param valid_y: validation labels
        """
        valid_X = np.array(valid_X).reshape(-1, self.n_features)
        valid_y = np.array(valid_y).reshape(-1)
        valid_pred = self.predict(valid_X)

        frontier = set()
        # for every leaf, examine its parent, if the parent's children are ALL LEAVES, it is a frontier
        for leaf in self.leaves:
            parent: ID3Node = leaf.parent
            if parent!=None and reduce(lambda x,y:x and y, [child.label_val!=None for child in parent.children.values()]):
                frontier.add(parent)
        frontier = list(frontier)
        while len(frontier) > 0:
            # pop up frontier firstly
            parent = frontier.pop(0)
            # get label if the pruning is taken
            unique_labels, label_counts = np.unique(self.y[parent.samples], return_counts=True)
            if_pruned_label = unique_labels[np.argmax(label_counts)]

            n_samples, n_unprune_right, n_prune_right = 0, 0, 0
            # currently, the parent is a frontier, examine its all child
            for child in parent.children.values():
                if child.predicts != None:
                    n_samples += len(child.predicts)
                    # get num of correctly classified samples if not prune, compare predicted label with gt-label
                    equals = [valid_pred[i]==valid_y[i] for i in child.predicts]
                    n_unprune_right += sum(equals)
                    # get num of correctly classified samples if prune, compare prior label with gt-label
                    equals = [valid_y[i]==if_pruned_label for i in child.predicts]
                    n_prune_right += equals
            if n_samples == 0:
                continue

            # if there are improvements, execute prune
            if n_prune_right > n_unprune_right:
                # update leaves
                for child in parent.children.values():
                    self.leaves.remove(child)
                self.leaves.append(parent)
                # update frontier node
                parent.children.clear()
                parent.label_val = if_pruned_label
                # get to frontier's parent, examine it's a new frontier or not
                parent = parent.parent
                if parent == None:
                    break
                if reduce(lambda x,y:x and y, [child.label_val!=None for child in parent.children.values()]):
                    frontier.append(parent)

    def __get_split_attr(self, sample_ids, attr_set):
        """
        :param sample_ids: ids of samples in current node to be splited
        :param attr_set: attributes remains to examin
        :return: split attribute
        """
        X, y = self.X[sample_ids], self.y[sample_ids]
        ent0 = ID3TreeClassifier.cal_entropy(y)
        entropy_gain_dict = {}
        for attr in attr_set:
            # get all possible a_v for attr(a)
            attr_vals, sample_count = np.unique(X[:,attr], return_counts=True)
            # for every attr_val, calculate entropy, and rescale
            entropy_gain_dict[attr] = ent0-sum([
                sample_count[i] * ID3TreeClassifier.cal_entropy(y[X[:,attr] == attr_vals[i]]) for i in range(len(attr_vals))
            ])/len(X)  # here, len(X) is the sample count in current node; sum of all sample_count -> len(X)
        split_attr = max(entropy_gain_dict.items(), key=lambda d:d[1])[0]  # return attr list according to entropy
        return split_attr


    @staticmethod
    def cal_entropy(subset_labels):
        """
        :param subset_labels: subset of sample labels whose attr(a) is a_v, including pos & neg samples
        :return: entropy
        """
        # return num of positive & negative labels
        neg_pos_count = np.unique(subset_labels, return_counts=True)[1].astype(float)
        percent = neg_pos_count/np.sum(neg_pos_count)
        return -np.sum(percent * np.log2(percent))  # neg sum of p*log(p)

if __name__ == '__main__':
    classifier = ID3TreeClassifier()
    X = [
        ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软黏"],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软黏"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑"],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软黏"],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软黏"],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑"],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软黏"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑"],
        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"],
    ]
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    classifier.fit(X,y)
    pred = classifier.predict(X)
    print(np.mean(pred==y))


