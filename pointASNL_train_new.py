import tf_sampling
import tf_grouping
from tf_interpolate import three_nn, three_interpolate
import warnings
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, ReLU, Softmax, LeakyReLU
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import dataset
import sys
sys.path.append('tf_ops/sampling')
sys.path.append('tf_ops/grouping')
sys.path.append('tf_ops/3d_interpolation')
sys.path.append('nearest_neighbors')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

# 以下三个函数来自黄仁朗学长


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = -2 * tf.matmul(src, tf.transpose(dst, (0, 2, 1)))
    dist += tf.expand_dims(tf.reduce_sum(src**2, -1), axis=-1)
    dist += tf.expand_dims(tf.reduce_sum(dst**2, -1), axis=-2)
    dist = tf.transpose(dist, (0, 2, 1))
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, _ = xyz.shape
    centroids = list()
    distance = tf.ones([B, N]) * 1e10
    farthest = tf.convert_to_tensor(np.random.randint(0, N, B), dtype=tf.int32)
    batch_indices = tf.convert_to_tensor(
        np.arange(B).reshape((-1, 1)), dtype=tf.int32)

    for _ in range(npoint):
        farthest = tf.expand_dims(farthest, axis=-1)
        centroids.append(farthest)
        farthest = tf.concat([batch_indices, farthest], axis=-1)
        farthest = tf.expand_dims(tf.gather_nd(xyz, farthest), axis=1)
        dist = tf.reduce_sum((xyz-farthest)**2, axis=-1)
        distance = tf.where(distance > dist, dist, distance)
        farthest = tf.argmax(distance, axis=1, output_type=tf.int32)
    centroids = tf.concat(centroids, axis=1)
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    radius = radius**2
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx = tf.math.top_k(-sqrdists, nsample)
    sqrdists = -group_idx.values  # [B, S, N]
    group_idx = group_idx.indices  # [B, S, N]
    group_idx = tf.where(sqrdists < radius, group_idx, group_idx[:, :, :1])
    return group_idx

# 以下为复现的pointASNL正文


def ASNLk_query(k, querys, supports):
    """
    Input:
        support_pts: points you have, B*N1*3
        query_pts: points you want to know the neighbour index, B*N2*3
        k: Number of neighbours in knn search
    Output: 
        neighbor_idx: neighboring points indexes, B*N2*k
    """
    dist = square_distance(querys, supports)
    result = tf.math.top_k(-dist, k)
    result = result.indices
#     result = np.array(result)
    result = tf.convert_to_tensor(result, tf.int32)
    return result


def sampling(npoint, points, feature=None):
    """
    Input:
        npoint: number of sampling, int32
        points: points clouds to be sampled, B*N*D
    Output:
        points_sampling: B*npoint*D
    """
    batch_size = points.get_shape()[0]
    fps_index = tf_sampling.farthest_point_sample(npoint, points)
    batch_indices = tf.tile(tf.reshape(
        tf.range(batch_size), (-1, 1, 1)), (1, npoint, 1))  # (batch_size,npoint,1)
    idx = tf.concat([batch_indices, tf.expand_dims(fps_index, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(points, idx)
    else:
        return tf.gather_nd(points, idx), tf.gather_nd(feature, idx)


def grouping(feature, K, src_xyz, query_xyz, use_xyz=True, use_knn=True, radius=0.2):
    '''
    Input:
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    Output:
    grouped_feature
    grouped_xyz
    idx
    '''
    batch_size = src_xyz.get_shape()[0]
    npoint = query_xyz.get_shape()[1]
    use_knn = True
    if use_knn:
        point_indices = ASNLk_query(K, src_xyz, query_xyz)
        # point_indices = tf.py_function(
        # ASNLk_query, [K, src_xyz, query_xyz], tf.int32)
        point_indices = tf.cast(point_indices, dtype=tf.int32)
        batch_indices = tf.tile(tf.reshape(
            tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
        idx = tf.concat(
            [batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        idx.set_shape([batch_size, npoint, K, 2])
        grouped_xyz = tf.gather_nd(src_xyz, idx)
    else:
        point_indices, _ = tf_grouping.query_ball_point(
            radius, K, src_xyz, query_xyz)
        grouped_xyz = tf_grouping.group_point(src_xyz, point_indices)

    grouped_feature = tf.gather_nd(feature, idx)
    if use_xyz:
        grouped_feature = tf.concat([grouped_xyz, grouped_feature], axis=-1)
    return grouped_xyz, grouped_feature, idx


class ASNLconv2d(tf.keras.layers.Layer):
    def __init__(self, hidden_units, kernelsize, padding='valid', strides=[1, 1], is_training=True):
        super().__init__()
        self.kernel_h, self.kernel_w = kernelsize
        self.stride_h, self.stride_w = strides
        self.conv = []
        self.num_conv = 0
        self.padding = padding
        self.istrain = is_training
        for i, num_hidden_unit in enumerate(hidden_units):
            out_channel = num_hidden_unit
            self.conv.append(tf.keras.layers.Conv2D(out_channel, [self.kernel_h, self.kernel_w], strides=[
                             self.stride_h, self.stride_w], padding=self.padding))
            self.num_conv += 1
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.leakyrelu = tf.keras.layers.LeakyReLU()

    def call(self, xyz, bn=False, actfunc='relu'):
        for i in range(self.num_conv):
            if i == 0:
                net = self.conv[i](xyz)
            else:
                net = self.conv[i](net)
        if bn == True:
            net = self.batchnorm(net)
        if actfunc is not None:
            if actfunc == 'relu':
                net = self.relu(net)
            elif actfunc == 'leaky_relu':
                net = self.leakyrelu(net)
        return net


class ASNLconv1d(tf.keras.layers.Layer):
    def __init__(self, hidden_units, kernelsize, padding='valid', strides=1, is_training=True):
        super().__init__()
        self.kernel = kernelsize
        self.padding = padding
        self.stride = strides
        self.istrain = is_training
        self.conv = []
        self.num_conv = 0
        for i, hidden_unit in enumerate(hidden_units):
            out_channel = hidden_unit
            self.conv.append(tf.keras.layers.Conv1D(
                out_channel, self.kernel, self.stride, self.padding))
            self.num_conv += 1
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.leakyrelu = tf.keras.layers.LeakyReLU()

    def call(self, xyz, bn=False, actfunc='relu'):
        for i in range(self.num_conv):
            if i == 0:
                net = self.conv[i](xyz)
            else:
                net = self.conv[i](net)
        if bn == True:
            net = self.batchnorm(net)
        if actfunc is not None:
            if actfunc == 'relu':
                net = self.relu(net)
            elif actfunc == 'leaky_relu':
                net = self.leakyrelu(net)
        return net


class SampleWeight(tf.keras.Model):
    def __init__(self, mlps, is_training, channel):
        super().__init__()
        self.mlps = mlps
        self.istrain = is_training
        self.channel = channel
        self.bottleneck_channel = max(32, channel//2)
        self.conv0 = ASNLconv2d([self.bottleneck_channel*2], [1, 1],
                                padding='valid', strides=[1, 1], is_training=is_training)
        self.conv1 = ASNLconv2d([self.bottleneck_channel], [
                                1, 1], padding='valid', strides=[1, 1], is_training=is_training)
        self.softmax0 = tf.keras.layers.Softmax(axis=-1)
        self.softmax1 = tf.keras.layers.Softmax(axis=2)
        self.conv = []
        self.num_conv = 0
        for i, mlp in enumerate(mlps):
            out_channel = mlp
            self.conv.append(ASNLconv2d(
                [mlp], [1, 1], 'valid', [1, 1], self.istrain))
            self.num_conv += 1

    def call(self, new_point, grouped_xyz, bn, actfunc, scaled=True):
        batch_size, npoint, nsample, channel = new_point.get_shape()
        normalized_xyz = grouped_xyz - \
            tf.tile(tf.expand_dims(grouped_xyz[:, :, 0, :], 2), [
                    1, 1, nsample, 1])
        new_point = tf.concat([normalized_xyz, new_point], axis=-1)
        transformed_feature = self.conv0(new_point, bn, actfunc)
        transformed_new_point = self.conv1(new_point, bn, actfunc)
        transformed_feature1 = transformed_feature[:,
                                                   :, :, :self.bottleneck_channel]
        feature = transformed_feature[:, :, :, self.bottleneck_channel:]

        weights = tf.matmul(transformed_new_point,
                            transformed_feature1, transpose_b=True)
        if scaled:
            weights = weights / \
                tf.sqrt(tf.cast(self.bottleneck_channel, tf.float32))
        weights = self.softmax0(weights)
        channel = self.bottleneck_channel

        new_group_features = tf.matmul(weights, feature)
        new_group_features = tf.reshape(
            new_group_features, (batch_size, npoint, nsample, channel))
        for i in range(self.num_conv):
            if i < self.num_conv-1:
                new_group_features = self.conv[i](
                    new_group_features, bn, actfunc)
            else:
                new_group_features = self.conv[i](new_group_features, bn, None)
        new_group_features = self.softmax1(new_group_features)
        return new_group_features


class AdaptiveSampling(tf.keras.Model):
    def __init__(self, num_neighbour, num_channel, is_training):
        super().__init__()
        self.num_neighbour = num_neighbour
        self.num_channel = num_channel
        self.istrain = is_training
        self.sampleweight = SampleWeight(
            [32, 1+self.num_channel], is_training=self.istrain, channel=self.num_channel)

    def call(self, x, y, bn, actfunc='relu'):
        nsample, num_channel = y.get_shape()[-2:]
        if self.num_neighbour == 0:
            new_xyz = x[:, :, 0, :]
            new_feature = y[:, :, 0, :]
            return new_xyz, new_feature
        shift_group_xyz = x[:, :, :self.num_neighbour, :]
        shift_group_feature = y[:, :, :self.num_neighbour, :]
        sample_weight = self.sampleweight(
            shift_group_feature, shift_group_xyz, bn, actfunc)
        new_weight_xyz = tf.tile(tf.expand_dims(
            sample_weight[:, :, :, 0], axis=-1), [1, 1, 1, 3])
        new_weight_feature = sample_weight[:, :, :, 1:]
        new_xyz = tf.reduce_sum(tf.multiply(
            shift_group_xyz, new_weight_xyz), axis=[2])
        new_feature = tf.reduce_sum(tf.multiply(
            shift_group_feature, new_weight_feature), axis=[2])
        return new_xyz, new_feature


class nonlocalcell(tf.keras.Model):
    def __init__(self, mlps, is_training, mode='dot'):
        super().__init__()
        self.mlps = mlps
        self.istrain = is_training
        self.mode = mode
        self.bottleneck_channel = self.mlps[0]
        self.conv0 = ASNLconv2d([self.bottleneck_channel*2], [1, 1],
                                padding='valid', strides=[1, 1], is_training=self.istrain)
        self.conv1 = ASNLconv2d([self.bottleneck_channel], [
                                1, 1], padding='valid', strides=[1, 1], is_training=self.istrain)
        self.conv2 = ASNLconv2d([1], [1, 1], padding='valid', strides=[
                                1, 1], is_training=self.istrain)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.conv3 = ASNLconv2d(
            [self.mlps[-1]], [1, 1], padding='valid', strides=[1, 1], is_training=self.istrain)

    def call(self, x, y, bn=True, actfunc='relu', scaled=True):
        feature = x
        new_point = y
        batch_size, npoint, nsample, channel = new_point.get_shape()
        ndataset = feature.get_shape()[1]
        feature = tf.expand_dims(feature, axis=2)
        transformed_feature = self.conv0(feature, bn, actfunc=None)
        transformed_new_point = self.conv1(new_point, bn, actfunc=None)
        transformed_new_point = tf.reshape(
            transformed_new_point, [batch_size, npoint*nsample, self.bottleneck_channel])
        transformed_feature1 = tf.squeeze(
            transformed_feature[:, :, :, :self.bottleneck_channel], axis=[2])
        transformed_feature2 = tf.squeeze(
            transformed_feature[:, :, :, self.bottleneck_channel:], axis=[2])
        if self.mode == 'dot':
            attention_map = tf.matmul(
                transformed_new_point, transformed_feature1, transpose_b=True)
            if scaled:
                attention_map = attention_map / \
                    tf.sqrt(tf.cast(self.bottleneck_channel, tf.float32))
        elif self.mode == 'concat':
            tile_transformed_feature1 = tf.tile(tf.expand_dims(
                transformed_feature1, axis=1), (1, npoint*nsample, 1, 1))
            tile_transformed_new_point = tf.tile(tf.reshape(transformed_new_point, (
                batch_size, npoint*nsample, 1, self.bottleneck_channel)), (1, 1, ndataset, 1))
            merged_feature = tf.concat(
                [tile_transformed_feature1, tile_transformed_new_point], axis=-1)
            attention_map = self.conv2(merged_feature, bn=bn)
            attention_map = tf.reshape(
                attention_map, (batch_size, npoint*nsample, ndataset))
        attention_map = self.softmax(attention_map)
        new_nonlocal_point = tf.matmul(attention_map, transformed_feature2)
        new_nonlocal_point = self.conv3(tf.reshape(new_nonlocal_point, [
                                        batch_size, npoint, nsample, self.bottleneck_channel]), bn=bn)
        new_nonlocal_point = tf.squeeze(new_nonlocal_point, axis=[1])
        return new_nonlocal_point


class ASNLabstraction(tf.keras.Model):
    def __init__(self, num_channel, npoint, nsample, mlps, is_training, use_knn=True, radius=True, neighbour=8, NL=True):
        super().__init__()
        self.num_channel = num_channel
        self.npoint = npoint
        self.nsample = nsample
        self.mlps = mlps
        self.istrain = is_training
        self.radius = radius
        self.neighbour = neighbour
        self.use_knn = use_knn
        self.NL = NL
        self.nl_channel = mlps[-1]
        self.adaptsample = AdaptiveSampling(
            num_neighbour=self.neighbour, num_channel=self.num_channel+3, is_training=is_training)
        self.nonlocalcell = nonlocalcell(
            mlps=[max(32, self.num_channel//2), self.nl_channel], is_training=self.istrain)
        self.conv0 = ASNLconv1d(
            [mlps[-1]], 1, padding='valid', strides=1, is_training=self.istrain)
        self.conv1 = ASNLconv2d(
            mlps[:-1], [1, 1], padding='valid', strides=[1, 1], is_training=self.istrain)
        self.conv2 = ASNLconv2d([32], [1, 1], padding='valid', strides=[
                                1, 1], is_training=self.istrain)
        self.conv3 = ASNLconv2d(
            [mlps[-1]], [1, mlps[-2]], padding='valid', strides=[1, 1], is_training=self.istrain)
        self.conv4 = ASNLconv1d(
            [mlps[-1]], 1, padding='valid', strides=1, is_training=self.istrain)

    def call(self, xyz, feature, bn=True):
        batch_size, num_points, num_channel = feature.get_shape()
        if num_points == self.npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(self.npoint, xyz, feature)
        grouped_xyz, new_point, idx = grouping(
            feature, self.nsample, xyz, new_xyz, use_knn=self.use_knn, radius=self.radius)
        if num_points != self.npoint:
            new_xyz, new_feature = self.adaptsample(
                grouped_xyz, new_point, bn=bn)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2),
                               [1, 1, self.nsample, 1])
        new_point = tf.concat([grouped_xyz, new_point], axis=-1)
        if self.NL:
            new_nonlocal_point = self.nonlocalcell(
                feature, tf.expand_dims(new_feature, axis=1), bn=bn)
        skip_space = tf.reduce_max(new_point, axis=[2])
        skip_space = self.conv0(skip_space, bn=bn)
        new_point = self.conv1(new_point, bn=bn)
        weight = self.conv2(grouped_xyz, bn=bn)
        new_point = tf.transpose(new_point, [0, 1, 3, 2])
        new_point = tf.matmul(new_point, weight)
        new_point = self.conv3(new_point, bn=bn)
        new_point = tf.squeeze(new_point, [2])
        new_point = tf.add(new_point, skip_space)
        if self.NL:
            new_point = tf.add(new_point, new_nonlocal_point)
        new_point = self.conv4(new_point, bn=bn)
        return new_xyz, new_point


class ASNLdecoding(tf.keras.Model):
    def __init__(self, num_channel, nsample, mlps, is_training, use_xyz, use_knn, radius, mode='concat', NL=False):
        super().__init__()
        self.num_channel = num_channel
        self.nsample = nsample
        self.mlps = mlps
        self.istrain = is_training
        self.use_xyz = use_xyz
        self.use_knn = use_knn
        self.radius = radius
        self.mode = mode
        self.NL = NL
        if self.use_xyz:
            self.channel = self.num_channel+3
        else:
            self.channel = self.num_channel
        self.nonlocalcell = nonlocalcell(
            [max(32, self.channel)], is_training=self.istrain, mode=self.mode)
        self.conv0 = ASNLconv2d([32], [1, 1], padding='valid', strides=[
                                1, 1], is_training=self.istrain)
        self.conv1 = ASNLconv2d([mlps[0]], [1, self.channel], padding='valid', strides=[
                                1, 1], is_training=self.istrain)
        self.conv2 = ASNLconv2d(mlps[1:], [1, 1], padding='valid', strides=[
                                1, 1], is_training=self.istrain)

    def call(self, xyz1, xyz2, point1, point2, bn):
        batch_size, num_point, num_channel = point2.get_shape()
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0/dist)/norm
        if self.NL:
            new_nonlocal_point = self.nonlocalcell(
                point1, tf.expand_dims(point2, axis=1), bn=bn)
            new_nonlocal_point = tf.squeeze(new_nonlocal_point, [1])
            point2 = tf.add(point2, new_nonlocal_point)
        interpolated_points = three_interpolate(point2, idx, weight)
        grouped_xyz, grouped_feature, idx = grouping(
            interpolated_points, self.nsample, xyz1, xyz1, use_xyz=self.use_xyz, use_knn=self.use_knn, radius=self.radius)
        grouped_xyz -= tf.tile(tf.expand_dims(xyz1, 2),
                               [1, 1, self.nsample, 1])
        weight = self.conv0(grouped_xyz, bn=bn)
        new_points = grouped_feature
        new_points = tf.transpose(new_points, [0, 1, 3, 2])
        new_points = tf.matmul(new_points, weight)
        new_points = self.conv1(new_points, bn=bn)
        if point1 is not None:
            new_point1 = tf.concat(
                axis=-1, values=[new_points, tf.expand_dims(point1, axis=2)])
        else:
            new_point1 = new_points
        new_point1 = self.conv2(new_point1, bn=bn)
        new_points = tf.squeeze(new_points, [2])
        return new_points


class ASNLpropogation(tf.keras.Model):
    def __init__(self, mlps, is_training):
        super().__init__()
        self.mlps = mlps
        self.istrain = is_training
        self.conv0 = ASNLconv2d(self.mlps, [1, 1], padding='valid', strides=[
                                1, 1], is_training=self.istrain)

    def call(self, xyz1, xyz2, point1, point2, bn):
        dist, index = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0/dist)/norm
        interpolated_points = three_interpolate(point2, index, weight)
        if point1 is not None:
            new_point1 = tf.concat(
                axis=2, values=[interpolated_points, point1])
        else:
            new_point1 = interpolated_points
        new_point1 = tf.expand_dims(new_point1, 2)
        new_point1 = self.conv0(new_point1, bn=bn)
        new_point1 = tf.squeeze(new_point1, [2])
        return new_point1


class set_model(tf.keras.Model):
    def __init__(self, num_points, num_class, is_training, num_channel=0):
        super().__init__()
        self.num_points = num_points
        self.num_point = [num_points//8, num_points //
                          32, num_points//128, num_points//256]
        self.num_class = num_class
        self.istrain = is_training
        self.num_channel = num_channel
        self.channel = num_channel+3
        self.abs0 = ASNLabstraction(self.channel, npoint=self.num_points, nsample=32, mlps=[
                                    16, 16, 32], is_training=self.istrain, neighbour=0, NL=False)
        self.abs1 = ASNLabstraction(32, npoint=self.num_point[0], nsample=32, mlps=[
                                    32, 32, 64], is_training=self.istrain, neighbour=8)
        self.abs2 = ASNLabstraction(32, npoint=self.num_point[0], nsample=32, mlps=[
                                    64, 64], is_training=self.istrain, neighbour=8, NL=False)
        self.abs3 = ASNLabstraction(64, npoint=self.num_point[1], nsample=32, mlps=[
                                    64, 64, 128], is_training=self.istrain, neighbour=4)
        self.abs4 = ASNLabstraction(64, npoint=self.num_point[1], nsample=32, mlps=[
                                    128, 128], is_training=self.istrain, neighbour=0, NL=False)
        self.abs5 = ASNLabstraction(128, npoint=self.num_point[2], nsample=32, mlps=[
                                    128, 128, 256], is_training=self.istrain, neighbour=0)
        self.abs6 = ASNLabstraction(128, npoint=self.num_point[2], nsample=32, mlps=[
                                    256, 256], is_training=self.istrain, neighbour=0, NL=False)
        self.abs7 = ASNLabstraction(256, npoint=self.num_point[3], nsample=32, mlps=[
                                    256, 256, 512], is_training=self.istrain, neighbour=0)
        self.abs8 = ASNLabstraction(512, npoint=self.num_point[3], nsample=32, mlps=[
                                    512, 512], is_training=self.istrain, neighbour=0, NL=False)
        self.pro0 = ASNLpropogation([512, 512], self.istrain)
        self.pro1 = ASNLpropogation([256, 256], self.istrain)
        self.pro2 = ASNLpropogation([256, 128], self.istrain)
        self.pro3 = ASNLpropogation([128, 128, 128], self.istrain)
        self.conv0 = ASNLconv1d(
            [128], 1, padding='valid', is_training=self.istrain, strides=1)
        self.drop = tf.keras.layers.Dropout(rate=0.5)
        self.conv1 = ASNLconv1d(
            [self.num_class], 1, padding='valid', is_training=self.istrain, strides=1)

    def call(self, point_cloud, bn):
        end_point = {}
        if self.num_channel > 0:
            l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
            l0_point = tf.slice(
                point_cloud, [0, 0, 3], [-1, -1, self.num_channel])
        else:
            l0_xyz = point_cloud
            l0_point = point_cloud
        end_point['l0_xyz'] = l0_xyz
        _, l0_point = self.abs0(l0_xyz, l0_point, bn)
        l1_xyz, l1_1_point = self.abs1(l0_xyz, l0_point, bn)
        _, l1_2_point = self.abs2(l0_xyz, l0_point, bn)
        l1_2_point += l1_1_point
        l2_xyz, l2_1_point = self.abs3(l1_xyz, l1_2_point, bn)
        _, l2_2_point = self.abs4(l2_xyz, l2_1_point, bn)
        l2_2_point += l2_1_point
        l3_xyz, l3_1_point = self.abs5(l2_xyz, l2_2_point, bn)
        _, l3_2_point = self.abs6(l3_xyz, l3_1_point, bn)
        l3_2_point += l3_1_point
        l4_xyz, l4_1_point = self.abs7(l3_xyz, l3_1_point, bn)
        _, l4_2_point = self.abs8(l4_xyz, l4_1_point, bn)
        l4_2_point += l4_1_point
        end_point['l1_xyz'] = l1_xyz
        l3_point = self.pro0(l3_xyz, l4_xyz, l3_2_point, l4_2_point, bn=True)
        l2_point = self.pro1(l2_xyz, l3_xyz, l2_2_point, l3_point, bn=True)
        l1_point = self.pro2(l1_xyz, l2_xyz, l1_2_point, l2_point, bn=True)
        l0_point = self.pro3(l0_xyz, l1_xyz, l0_point, l1_point, bn=True)
        net = self.conv0(l0_point, bn=True, actfunc='leaky_relu')
        end_point['feats'] = net
        net = self.drop(net)
        net = self.conv1(net, actfunc=None)
        return net, end_point


def repulsion_loss(pred, nsample=20, radius=0.07):
    """
    input:
    pred:(batch_size,npoint,3)
    output:
    loss
    """
    idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, pred, pred)
#     tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = tf_grouping.group_point(
        pred, idx)  # (batch_size,npoint,nsample,3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred**2, axis=-1)
    dist_square, idx = tf.math.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]
    dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def get_loss(batch_data, label, smpw=1.0, uniform_weight=0.1, radius=0.07):
    """
    pred: BxNxC,
    label: BxN,
    smpw: BxN
    """
    output, endpoint = model(batch_data, True)
    y = np.zeros(output.shape)
    for i in range(len(label)):
        a = label[i].numpy().astype(np.int32)
        y[i, a] = 1
    y = tf.convert_to_tensor(y, tf.float32)
    classify_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=output)
#     classify_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=output)
    uniform_loss = repulsion_loss(
        endpoint['l1_xyz'], nsample=20, radius=radius)
#     classify_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y,logits=output,pos_weight=smpw)
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss')
    total_loss = classify_loss_mean + uniform_weight * uniform_loss
    return total_loss


if __name__ = '__main__':
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001)
    batch_size = 8
    epoches = 50
    model = set_model(num_points=8192, num_class=20,
                      is_training=True, num_channel=0)

    def training():
        data = dataset.Semantickitti(
            root='/remote-home/share/ums_huangrenlang/semantic_kitti/data_odometry_velodyne/dataset', sample_points=8192, split='train')
        idxs = np.arange(0, len(data))
        batch_num = int(len(data)/batch_size)
        batch_num = 100
        print(batch_num)
        for i in range(epoches):
            loss_batch = 0
            for batch_idx in range(batch_num):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx+1) * batch_size
                batch_data, batch_label, batch_weight = dataset.batch_data(
                    data, idxs, start_idx, end_idx, 8192, 0)
                batch_data = tf.convert_to_tensor(batch_data, tf.float32)
                batch_label = tf.convert_to_tensor(batch_label, tf.float32)
                batch_weight = tf.convert_to_tensor(batch_weight, tf.float32)
                with tf.GradientTape() as tape:
                    allloss = get_loss(batch_data, batch_label, batch_weight)
                grads = tape.gradient(allloss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                loss_batch += allloss
            print(loss_batch)
    training()
