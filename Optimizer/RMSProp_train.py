from abc import ABC
import tensorflow as tf
import re


class RMSProp(tf.train.Optimizer, ABC):

    def __init__(self,
                 learning_rate=0.0001,
                 rho=0.9,
                 name='custom_rmsprop',
                 eps=0.000001,
                 **kwargs):
        super(RMSProp, self).__init__(False, name)
        self.learning_rate = kwargs.get("lr", learning_rate)  # handle lr=learning_rate
        self.rho = rho
        self.eps = eps

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            g_sq = tf.get_variable(
                name=param_name + "/rmsprop_g_sq",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            #  update of squared gradient
            next_g_sq = (
                    tf.multiply(self.rho, g_sq) + tf.multiply(1.0 - self.rho, grad ** 2))

            #  calculate the update based on grad and squared gradient
            update = grad / (tf.sqrt(next_g_sq) + self.eps)

            # scale the update by learning_rate
            update_with_lr = self.learning_rate * update

            #  update model parameter
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 g_sq.assign(next_g_sq)])
        return tf.group(*assignments, name=name)

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
