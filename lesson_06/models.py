import numpy as np


class Node:
    
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch
    
        
class Leaf_clsf:
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()
        

    def predict(self):
        '''Небольшие изменения метода, используются методы библиотеки numpy.'''
        # получение классов и количество их вхождений
        classes, counts = np.unique(self.labels, return_counts=True)
        
        # получение индекса максимального значения из counts
        idx = np.argmax(counts)
        return classes[idx]
    

class Leaf_regr:
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.prediction = self.predict()
        
        
    def predict(self):
        return self.targets.mean()


class BaseTree:
    
    def __init__(self, min_samples_leaf, max_depth, max_leafs, leaf_class):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.tree = None
        self.Leaf = leaf_class

    def _gain(self, left_labels, right_labels, root):

        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

        return root - p * self._criterion(left_labels) - (1 - p) * self._criterion(right_labels)

    def _split(self, data, labels, column_index, t):

        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    def _find_best_split(self, data, labels):

        root = self._criterion(labels)

        best_gain = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        for index in range(n_features):
            t_values = np.unique(data[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self._split(data, labels, index, t)
                #  пропуск разбиений, в которых в узле остается менее 5 объектов
                if len(true_data) < self.min_samples_leaf or len(false_data) < self.min_samples_leaf:
                    continue

                current_gain = self._gain(true_labels, false_labels, root)
                #  порог, на котором получается максимальный прирост качества
                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index

    def _build_tree(self, data, labels, current_depth=None):

        gain, t, index = self._find_best_split(data, labels)

        # возвращает лист, если: 
        # нет прироста информативности 
        # или по достижению максимальной глубины
        # или если кончились листья
        if gain == 0 or current_depth == self.max_depth or self.max_leafs == 0:
            return self.Leaf(data, labels)

        if current_depth is not None:
            current_depth += 1
            
        true_data, false_data, true_labels, false_labels = self._split(data, labels, index, t)

        # перед построением новых веток отнимаем лист от максимального количества
        # таким образом построение новых веток будет осуществляться с учетом оставшегося количества листьев
        # при каждом разбиении минимальное количество листьев (при текущей схеме дерева) увеличивается на 1, 
        # т.к. текущий узел стал нодой, а не листом 
        if self.max_leafs is not None:
            self.max_leafs -= 1
        
        true_branch = self._build_tree(true_data, true_labels, current_depth=current_depth)

        false_branch = self._build_tree(false_data, false_labels, current_depth=current_depth)

        return Node(index, t, true_branch, false_branch)

    def _classify_object(self, obj, node):
        
        if isinstance(node, self.Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self._classify_object(obj, node.true_branch)
        else:
            return self._classify_object(obj, node.false_branch)
        
    def fit(self, data, labels):
        
        # перед первым построением отнимаем лист от максимального количества
        # т.к. при вызове метода build_tree появится как минимум один лист
        if self.max_leafs is not None:
            self.max_leafs -= 1
        self.tree = self._build_tree(data, labels, current_depth=0)
        return self
        
    def predict(self, data):
        
        classes = []
        for obj in data:
            prediction = self._classify_object(obj, self.tree)
            classes.append(prediction)
        return np.array(classes)



class ClassificationTree(BaseTree):
    
    def __init__(self, min_samples_leaf=5, max_depth=None, max_leafs=None, leaf_class=Leaf_clsf):
        super().__init__(min_samples_leaf=min_samples_leaf, 
                         max_depth=max_depth, 
                         max_leafs=max_leafs, 
                         leaf_class=Leaf_clsf)
        
    def _criterion(self, labels):
        '''В качестве критерия - энтропия Шэннона'''
        # получение количества вхождений для каждого класса
        _, counts = np.unique(labels, return_counts=True)
        
        # вероятность каждого из классов
        p = counts/np.sum(counts)
        
        # расчет энтропии
        enthropy = - np.sum(p * np.log2(p))

        return enthropy



class RegressionTree(BaseTree):
    
    def __init__(self, min_samples_leaf=5, max_depth=None, max_leafs=None, leaf_class=Leaf_regr):
        super().__init__(min_samples_leaf=min_samples_leaf, 
                         max_depth=max_depth, 
                         max_leafs=max_leafs, 
                         leaf_class=Leaf_regr)
        
    def _criterion(self, targets):
        return np.mean((targets - np.mean(targets))**2)
