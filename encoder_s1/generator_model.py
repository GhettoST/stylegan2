import math
import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


# 定义变量learnable_dlatents（batch_size, 18, 512），用于训练
def create_variable_for_generator(name, batch_size, tiled_dlatent, model_scale=18):
    if tiled_dlatent:
        low_dim_dlatent = tf.get_variable('learnable_dlatents',
                                          shape=(batch_size, 512),
                                          dtype='float32',
                                          initializer=tf.initializers.random_normal())
        return tf.tile(tf.expand_dims(low_dim_dlatent, axis=1), [1, model_scale, 1])
    else:
        return tf.get_variable('learnable_dlatents',
                               shape=(batch_size, model_scale, 512),
                               dtype='float32',
                               initializer=tf.initializers.random_normal())


# StyleGAN生成器，生成batch_size个图片，用于训练模型
class Generator:
    # 初始化
    def __init__(self, model, batch_size, clipping_threshold=2, tiled_dlatent=False, model_res=1024,
                 randomize_noise=False):
        self.batch_size = batch_size
        self.tiled_dlatent = tiled_dlatent
        self.model_scale = int(2 * (math.log(model_res, 2) - 1))  # For example, 1024 -> 18

        # 初始张量为全0（batch_size, 512），通过create_variable_for_generator自定义输入：learnable_dlatents
        # functools.partial为偏函数
        if tiled_dlatent:
            self.initial_dlatents = np.zeros((self.batch_size, 512))
            model.components.synthesis.run(np.zeros((self.batch_size, self.model_scale, 512)),
                                           randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                                           custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size,
                                                                  tiled_dlatent=True),
                                                          partial(create_stub, batch_size=batch_size)],
                                           structure='fixed')
        # 初始张量为全0（batch_size, 18, 512），通过create_variable_for_generator自定义输入：learnable_dlatents
        else:
            self.initial_dlatents = np.zeros((self.batch_size, self.model_scale, 512))
            model.components.synthesis.run(self.initial_dlatents,
                                           randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                                           custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size,
                                                                  tiled_dlatent=False, model_scale=self.model_scale),
                                                          partial(create_stub, batch_size=batch_size)],
                                           structure='fixed')

        self.dlatent_avg_def = model.get_var(
            'dlatent_avg')  # Decay for tracking the moving average of W during training. None = disable.
        self.reset_dlatent_avg()
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        # 定义dlatent_variable，遍历全局变量的名字空间找到learnable_dlatents
        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        # 定义全零的初始向量，若没有在外部指定训练的初始dlatents，则使用全零向量开始寻找ref_images的dlatents最优解（这估计会很慢，而且不容易收敛）
        self.set_dlatents(self.initial_dlatents)

        def get_tensor(name):
            try:
                return self.graph.get_tensor_by_name(name)
            except KeyError:
                return None

        # 定义输出
        self.generator_output = get_tensor('G_synthesis_1/_Run/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat_1/concat:0')
        # If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis/_Run/concat_1/concat:0')
        if self.generator_output is None:
            for op in self.graph.get_operations():
                print(op)
            raise Exception("Couldn't find G_synthesis_1/_Run/concat tensor output")
        # 定义方法，将输出的张量转换为图片
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        # 设定区间[-2, +2]
        clipping_mask = tf.math.logical_or(self.dlatent_variable > clipping_threshold,
                                           self.dlatent_variable < -clipping_threshold)
        # 以dlatent_variable为均值，按正态分布取值，并赋值给clipped_values
        clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape),
                                  self.dlatent_variable)
        # 将clipped_values赋值给神经网络图中的变量dlatent_variable，构建优化迭代的反馈输入
        self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)

    # 归零
    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    # 设置训练开始时的dlatents初始值，将shape统一调整为（batch_size, 512）或者（batch_size, model_scale, 512）
    # 将dlatents作为初始值赋给dlatent_variable
    def set_dlatents(self, dlatents):
        if self.tiled_dlatent:
            if (dlatents.shape != (self.batch_size, 512)) and (dlatents.shape[1] != 512):
                dlatents = np.mean(dlatents, axis=1)
            if (dlatents.shape != (self.batch_size, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size - dlatents.shape[0], 512))])
            assert (dlatents.shape == (self.batch_size, 512))
        else:
            if (dlatents.shape[1] > self.model_scale):
                dlatents = dlatents[:, :self.model_scale, :]
            if (dlatents.shape != (self.batch_size, self.model_scale, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size - dlatents.shape[0], self.model_scale, 512))])
            assert (dlatents.shape == (self.batch_size, self.model_scale, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents))

    # 对dlatent_variable执行stochastic_clip操作
    def stochastic_clip_dlatents(self):
        self.sess.run(self.stochastic_clip_op)

    # 读取dlatent_variable
    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def get_dlatent_avg(self):
        return self.dlatent_avg

    def set_dlatent_avg(self, dlatent_avg):
        self.dlatent_avg = dlatent_avg

    def reset_dlatent_avg(self):
        self.dlatent_avg = self.dlatent_avg_def

    # 用dlatents生成图片
    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)

