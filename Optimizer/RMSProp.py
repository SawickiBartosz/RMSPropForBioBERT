from tensorflow.math import sqrt
from tensorflow import function
from tensorflow.keras.optimizers import Optimizer


# resource: https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb
class RMSProp(Optimizer):

    def __init__(self, beta=0.9,
                 learning_rate=0.0001,
                 name='custom_rmsprop',
                 eps=0.000001,
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))  # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta", beta)
        self._set_hyper("eps", eps)

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "Eg_sq")

    @function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        Eg_sq = self.get_slot(var, "Eg_sq")
        beta = self._get_hyper("beta", var_dtype)
        eps = self._get_hyper("eps", var_dtype)
        Eg_sq.assign(beta * Eg_sq + (1.0 - beta) * grad ** 2)
        var.assign_sub(lr_t * grad / (sqrt(Eg_sq) + eps))

    def _resource_apply_sparse(self, grad, var):
        # To be implemented if needed
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter(self._learning_rate),
            "beta": self._serialize_hyperparameter(self.beta)
        }
