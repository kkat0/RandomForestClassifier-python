import numpy as np
from queue import Queue

class DecisionTreeNode:
    def __init__(self):
        self.n = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.info_gain = None
        self.depth = None
        self.label = None

    def build(self, data, target, depth, max_depth):
        self.n = len(data)
        self.depth = depth

        classes = np.unique(target, return_counts=True)

        self.label = classes[0][np.argmax(classes[1])]

        if len(classes[0]) == 1 or (max_depth != None and max_depth <= self.depth):
            return

        n_features = data.shape[1]
        max_features = int(np.sqrt(n_features))

        self.info_gain = 0.0
        better_split_found = False

        # for f in range(n_features):
        for f in np.random.permutation(n_features)[:max_features]:
            uniq_feature = np.unique(data[:, f])
            split_point = (uniq_feature[:-1] + uniq_feature[1:]) / 2

            for threshold in split_point:
                target_left = target[data[:, f] < threshold]
                target_right = target[data[:, f] >= threshold]

                if len(target_left) == 0 or len(target_right) == 0:
                    continue

                g = self.calc_info_gain(target, target_left, target_right)

                if self.info_gain < g:
                    better_split_found = True
                    self.info_gain = g
                    self.feature = f
                    self.threshold = threshold

        if not better_split_found:
            return

        data_left = data[data[:, self.feature] < self.threshold]
        target_left = target[data[:, self.feature] < self.threshold]
        self.left = DecisionTreeNode()
        self.left.build(data_left, target_left, depth + 1, max_depth)

        data_right = data[data[:, self.feature] >= self.threshold]
        target_right = target[data[:, self.feature] >= self.threshold]
        self.right = DecisionTreeNode()
        self.right.build(data_right, target_right, depth + 1, max_depth)

    def calc_info_gain(self, target, target_left, target_right):
        if self.impurity == None:
            self.impurity = self.calc_gini(target)

        n_left = len(target_left)
        n_right = len(target_right)

        impurity_left = self.calc_gini(target_left)
        impurity_right = self.calc_gini(target_right)

        return self.impurity - (n_left / self.n) * impurity_left - (n_right / self.n) * impurity_right

    def predict(self, data):
        if self.feature != None:
            if data[self.feature] < self.threshold:
                return self.left.predict(data)
            else:
                return self.right.predict(data)

        return self.label
    
    def calc_gini(self, target):
        n = target.shape[0]
        freq = np.unique(target, return_counts=True)[1]

        gini = 1.0 - ((freq / n) ** 2).sum()

        return gini


class DecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.dim_data = None
        self.feature_importance = None
        self.max_depth = max_depth

    def fit(self, data, target):
        self.root = DecisionTreeNode()
        self.root.build(data, target, 0, self.max_depth)
        self.dim_data = data.shape[1]

    def predict(self, data):
        assert(self.dim_data == data.shape[1])

        if self.root == None:
            raise Exception("Not trained yet.")

        predictions = []
        for d in data:
            predictions.append(self.root.predict(d))

        return np.array(predictions)

    def calc_feature_importances(self):
        if self.root == None:
            raise Exception("Not trained yet.")

        importances = np.zeros(self.dim_data)

        queue = Queue()
        queue.put(self.root)

        while not queue.empty():
            node = queue.get()

            if node.feature == None:
                continue

            importances[node.feature] += node.n / self.root.n * node.info_gain

            queue.put(node.left)
            queue.put(node.right)

        importances /= importances.sum()

        return importances


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.d_trees = None
        self.dim_data = None

    def fit(self, data, target):
        self.d_trees = np.array([DecisionTree(max_depth=self.max_depth) for _ in range(self.n_trees)])
        self.dim_data = data.shape[1]

        data, target = self.bootstrap_sampling(data, target)

        for i in range(self.n_trees):
            self.d_trees[i].fit(data[i], target[i])
            print("progress: {}%".format((i + 1) / self.n_trees * 100))

    def predict(self, data):
        assert(self.dim_data == data.shape[1])

        if self.dim_data == None:
            raise Exception("Not trained yet.")

        predictions = np.array([d_tree.predict(data) for d_tree in self.d_trees]).T
        ret = np.ndarray(predictions.shape[0], dtype=int)

        for i, pre in enumerate(predictions):
            votes = np.unique(pre, return_counts=True)
            ret[i] = votes[0][np.argmax(votes[1])]

        return ret

    def calc_feature_importances(self):
        if self.dim_data == None:
            raise Exception("Not trained yet.")

        importances = np.zeros(self.dim_data)

        for i in range(self.n_trees):
            importances += self.d_trees[i].calc_feature_importances()
        
        importances /= self.n_trees
        importances /= importances.sum()

        return importances
    
    def bootstrap_sampling(self, data, target):
        bags_data = []
        bags_target = []

        for _ in range(self.n_trees):
            idx = np.random.randint(0, data.shape[0], data.shape[0])
            bags_data.append(data[idx])
            bags_target.append(target[idx])
        
        return np.array(bags_data), np.array(bags_target)
