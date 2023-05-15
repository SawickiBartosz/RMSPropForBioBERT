from tensorflow.math import sqrt
from keras.optimizers import optimizer


class RMSProp(optimizer.Optimizer):

    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self._built = False
        self.var_list = None
        self.Eg_sq = None
        self._learning_rate = 0.001
        self.beta = beta

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
        pass
