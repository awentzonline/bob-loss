from keras import backend as K
from keras.layers import InputSpec, Layer


class RepeatVectorND(Layer):
    def __init__(self, num_repeats=1, axis=1, **kwargs):
        self.num_repeats = num_repeats
        self.axis = axis
        self.input_spec = [InputSpec(ndim='2+')]
        super(RepeatVectorND, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return K.repeat_elements(
            K.expand_dims(x, self.axis), self.num_repeats, self.axis
        )

    def get_output_shape_for(self, input_shape):
        print input_shape, input_shape[:self.axis] + (self.num_repeats,) + input_shape[self.axis:]
        return input_shape[:self.axis] + (self.num_repeats,) + input_shape[self.axis:]

    def get_config(self):
        config = {
            'num_repeats': self.num_repeats,
            'axis': self.axis
        }
        base_config = super(RepeatVectorND, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
