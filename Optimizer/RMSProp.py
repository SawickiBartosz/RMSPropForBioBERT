from tensorflow.math import sqrt
from keras.optimizers import optimizer


class RMSProp(optimizer.Optimizer):

    def __init__(self, beta=0.9,
                 learning_rate=0.0001,
                 name='custom_rmsprop',
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 jit_compile=True,
                 **kwargs):
        super().__init__(name=name,
                         weight_decay=weight_decay,
                         clipnorm=clipnorm,
                         clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm,
                         use_ema=use_ema,
                         ema_momentum=ema_momentum,
                         ema_overwrite_frequency=ema_overwrite_frequency,
                         jit_compile=jit_compile,
                         **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta = beta

        self._built = False
        self.var_list = None
        self.Eg_sq = None

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self.Eg_sq = []
        for var in var_list:
            self.Eg_sq.append(self.add_variable_from_reference(model_variable=var, variable_name="Eg_sq"))
        self.var_list = var_list

    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        Eg = self.Eg_sq[self._index_dict[var_key]]
        Eg.assign(self.beta * Eg + (1 - self.beta) * gradient**2)
        variable.assign_sub((self.learning_rate * gradient) / sqrt(Eg))

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter(
                self._learning_rate
            ),
            "beta": self.beta
        })
        return config
