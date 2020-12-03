import os
import numpy as np
import skimage.io as skio

from pipeline import Sand, SandHeap


def load_data(work_path):
    """
    加载一个路径的CT照片生成一个大的数组容器（应该是二值化的图像），并做好联通区域标注。

    label(input, structure=None, output=None)
    Label features in an array.
    --------
    input: array_like
        An array-like object to be labeled. Any non-zero values in `input` are
        counted as features and zero values are considered the background.
    structure: array_like, optional
        An SE that defines feature connections. It must be centrosymmetric.
        If no SE is provided, one is automatically generated with a squared
        connectivity equal to one.
    --------
    Returns: label: ndarray or int, num_features: int
        label: An integer ndarray where each unique feature in `input` has a
            unique label in the returned array.
        num_features: How many objects were found. 
    --------
    If `output` is None, this function returns a tuple of (labeled_array, 
    num_features).
    If `output` is a ndarray, then it will be updated with values in `labeled_
    array` and only `num_features` will be returned by this function.
    """
    imgs = os.listdir(work_path)
    imgs.sort()
    nimgs = len(imgs)
    h, w = skio.imread(os.path.join(work_path, imgs[0])).shape
    data = np.empty((nimgs, h, w), dtype=np.uint8)
    for i in range(nimgs):
        img = skio.imread(os.path.join(work_path, imgs[i]))
        data[i] = img
    data = data[:, 100:1948, 100:1948]  # 为啥是100:1948???
    return data


def get_learn_data(data):
    """Get the data that will be input to the nerual network.
        用64*64*64的cube封装单个颗粒后，得到一个个单独的颗粒的集合
    其中x表示原始位姿加9次随机旋转后的颗粒集，y表示10次位姿归一化的颗粒集
    也就是说，x[i]至x[i+10]其实是同一个颗粒只不过位姿不同；y[i]至y[i+10]
    也是一个颗粒，只不过进行了位姿归一化，并且这10个颗粒位姿相同，意味着它
    们完全相同（尚不知这样的意义是什么）。"""

    heap = SandHeap(data)
    labels = range(1, heap.num + 1)
    original = np.empty((heap.num*10, 1, heap.cube_size, heap.cube_size, heap.cube_size),
                        dtype=np.uint8)
    normalized = np.empty((heap.num*10, 1, heap.cube_size, heap.cube_size, heap.cube_size),
                          dtype=np.uint8)
    for label in labels:
        try:
            cube = heap.get_cube(label)
            sand = Sand(cube)
        except:
            print("Something wrong with this label: {}".format(label))
            continue
        cube_n = sand.pose_normalization()
        original[10 * (label-1), 1] = cube
        normalized[10 * (label-1), 1] = cube_n
        for i in range(1, 10):
            original[10*(label-1) + i, 1] = sand.random_rotate()
            normalized[10*(label-1) + i, 1] = cube_n
    return original, normalized

class DataSet(object):  # ？？？
    """可能跟tensorflow的批量训练习惯有关"""

    def __init__(self, stones, labels):
        self._stones = stones
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(stones)
        self.perm = np.random.permutation(self._num_examples)  # 随机排序

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle (v./n.洗牌;把...变换位置) the data
            np.random.shuffle(self.perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._stones[self.perm[start:end]], self._labels[self.perm[start:end]]