import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.python.distribute import parameter_server_strategy


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # with tf.GradientTape() as tape:
        #     y_pred = self(x, training=True)  # Forward pass
        #     # Compute the loss value
        #     # (the loss function is configured in `compile()`)
        #     loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}
        
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            # y_pred = get_peaks(y_pred)
            # y = get_peaks(y)
            # y, y_pred = filt_peaks(y, y_pred)
            # y = tf.cast(y, tf.float32)
            # y_pred = tf.cast(y_pred, tf.float32)

            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
            # For custom training steps, users can just write:
            #   trainable_variables = self.trainable_variables
            #   gradients = tape.gradient(loss, trainable_variables)
            #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            # The _minimize call does a few extra steps unnecessary in most cases,
            # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                    self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


def _minimize(strategy, tape, optimizer, loss, trainable_variables):
  """Minimizes loss for one step by updating `trainable_variables`.

  This is roughly equivalent to

  ```python
  gradients = tape.gradient(loss, trainable_variables)
  self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  ```

  However, this function also applies gradient clipping and loss scaling if the
  optimizer is a LossScaleOptimizer.

  Args:
    strategy: `tf.distribute.Strategy`.
    tape: A gradient tape. The loss must have been computed under this tape.
    optimizer: The optimizer used to minimize the loss.
    loss: The loss tensor.
    trainable_variables: The variables that will be updated in order to minimize
      the loss.
  """

  with tape:
    if isinstance(optimizer, lso.LossScaleOptimizer):
      loss = optimizer.get_scaled_loss(loss)

  gradients = tape.gradient(loss, trainable_variables)

  # Whether to aggregate gradients outside of optimizer. This requires support
  # of the optimizer and doesn't work with ParameterServerStrategy and
  # CentralStroageStrategy.
  aggregate_grads_outside_optimizer = (
      optimizer._HAS_AGGREGATE_GRAD and  # pylint: disable=protected-access
      not isinstance(strategy.extended,
                     parameter_server_strategy.ParameterServerStrategyExtended))

  if aggregate_grads_outside_optimizer:
    # We aggregate gradients before unscaling them, in case a subclass of
    # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
    # done on scaled gradients, not unscaled gradients, for numeric stability.
    gradients = optimizer._aggregate_gradients(zip(gradients,  # pylint: disable=protected-access
                                                   trainable_variables))
  if isinstance(optimizer, lso.LossScaleOptimizer):
    gradients = optimizer.get_unscaled_gradients(gradients)
  gradients = optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
  if trainable_variables:
    if aggregate_grads_outside_optimizer:
      optimizer.apply_gradients(
          zip(gradients, trainable_variables),
          experimental_aggregate_gradients=False)
    else:
      optimizer.apply_gradients(zip(gradients, trainable_variables))

@tf.function
def get_peaks(y):
    # y: (N,)
    data_reshaped = tf.reshape(y, (1, -1, 1)) # (1, N, 1)
    max_pooled_in_tensor =  tf.nn.max_pool(data_reshaped, (20,), 1,'SAME')
    maxima = tf.equal(data_reshaped,max_pooled_in_tensor) # (1, N, 1)
    maxima = tf.cast(maxima, tf.float32)
    maxima = tf.squeeze(maxima) # (N,1)
    peaks = tf.where(maxima) # now only the Peak Indices (A, 3)
    peaks = tf.reshape(peaks, (tf.size(y),)) # (A,1)

    return peaks

# x: true y: prediction
# input: peaks of truth and prediction as tensor...
@tf.function
def filt_peaks(x,y):
    def true_fn():
        return min
    def false_fn():
        return tf.cast(-1, tf.int64)
    max_offset = 10
    mask = tf.cast(tf.zeros(tf.size(x)),tf.bool) # tensor with size of x (truth data)
    # check which peaks of truth are recognized in pred
    min = 0
    min = tf.cast(min, tf.int64)

    # for item in y: # items of predicion
    #     diff = tf.abs(x - item) # diff of truth data and item
    #     min = tf.reduce_min(diff) # minimum of diff
    #     min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    #     temp_mask = tf.equal(min, diff)
    #     mask = tf.logical_or(mask, temp_mask)

    # x = tf.boolean_mask(x, mask)
    def fn(item):
        def true_fn():
            return tf.cast(min, tf.float64)
        def false_fn():
            return tf.cast(-1, tf.float64)
        diff = tf.abs(x - item) # diff of truth data and item
        diff = tf.cast(diff, tf.float64)
        min = tf.reduce_min(diff) # minimum of diff
        min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
        temp_mask = tf.equal(min, diff)
        return temp_mask
    mask1 = tf.map_fn(fn=lambda item: fn(item), elems=y, fn_output_signature=tf.bool)
    mask1 = tf.reduce_any(mask1, 0)
    x = tf.boolean_mask(x,mask1)

    # check if outliners are in pred
    # mask = tf.cast(tf.zeros(tf.size(y)), tf.bool)
    # for item in x: 
    #     diff = tf.abs(y - item) # diff of truth data and item
    #     min = tf.reduce_min(diff) # minimum of diff
    #     min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
    #     temp_mask = tf.equal(min, diff)
    #     mask = tf.logical_or(mask, temp_mask)
    # y = tf.boolean_mask(y,mask)

    def fn2(item):
        def true_fn():
            return tf.cast(min, tf.float64)
        def false_fn():
            return tf.cast(-1, dtype=tf.float64)
        diff = tf.abs(y - item) # diff of truth data and item
        diff = tf.cast(diff, tf.float64)
        min = tf.reduce_min(diff) # minimum of diff
        min = tf.cond(tf.less(min, max_offset), true_fn, false_fn)
        temp_mask = tf.equal(min, diff)
        return temp_mask
    mask2 = tf.map_fn(fn=lambda item: fn2(item), elems=x, fn_output_signature=tf.bool)
    mask2 = tf.reduce_any(mask2, 0)
    y = tf.boolean_mask(y,mask2)

    return x, y 