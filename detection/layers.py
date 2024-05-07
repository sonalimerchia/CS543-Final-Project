import tensorflow as tf  

class SelectLayer(tf.keras.layers.Layer): 
    def __init__(self, index, **kwargs): 
        super(SelectLayer, self).__init__(**kwargs)
        self.index = index
        pass

    def call(self, inputs): 
        return inputs[..., self.index]

class ClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_val, max_val, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_val
        self.max_value = max_val
        pass

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)
    
class FloorLayer(tf.keras.layers.Layer): 
    def __init__(self, cast_dtype=tf.float32, **kwargs):
        super(FloorLayer, self).__init__(**kwargs)
        self.cast_dtype = cast_dtype
        pass

    def call(self, inputs):
        floor = tf.math.floor(inputs)
        return tf.cast(floor, self.cast_dtype)
    
class CeilLayer(tf.keras.layers.Layer): 
    def __init__(self, cast_dtype=tf.float32, **kwargs):
        super(CeilLayer, self).__init__(**kwargs)
        self.cast_dtype = cast_dtype
        pass

    def call(self, inputs):
        ceil = tf.math.ceil(inputs)

        if self.dtype is not None: 
            return tf.cast(ceil, self.cast_dtype)
        return ceil
    
class GatherLayer(tf.keras.layers.Layer): 
    def __init__(self, **kwargs): 
        super(GatherLayer, self).__init__(**kwargs)
        pass 

    def call(self, inputs): 
        return tf.gather(inputs[0], inputs[1], batch_dims=1)
    
class MultiplyScalarLayer(tf.keras.layers.Layer): 
    def __init__(self, scalar, **kwargs): 
        super(MultiplyScalarLayer, self).__init__(**kwargs)
        self.scalar = scalar
        pass

    def call(self, inputs): 
        return tf.multiply(self.scalar, inputs)
    
class SplitLayer(tf.keras.layers.Layer): 
    def __init__(self, end, **kwargs): 
        super(SplitLayer, self).__init__(**kwargs)
        self.end = end 
        pass 

    def call(self, inputs): 
        return inputs[..., :self.end]
    
class ValueLayer(tf.keras.layers.Layer): 
    def __init__(self, **kwargs): 
        super(ValueLayer, self).__init__(**kwargs)
        self.one = tf.constant(1, dtype=tf.float32)
        pass 

    def call(self, inputs): 
        alpha = inputs[0]
        left = inputs[1]
        right = inputs[2]
        
        sub1 = tf.subtract(self.one, alpha)
        prod1 = tf.multiply(sub1, left)
        prod2 = tf.multiply(alpha, right)
        return tf.add(prod1, prod2)
    
class TransposeLayer(tf.keras.layers.Layer): 
    def __init__(self, order, **kwargs): 
        super(TransposeLayer, self).__init__(**kwargs)
        self.order = order

    def call(self, inputs): 
        return tf.transpose(inputs, self.order)