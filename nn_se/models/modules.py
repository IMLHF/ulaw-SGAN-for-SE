import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..utils import complex_value_ops as c_ops
from .complex_value_layers import ComplexValueConv2d
from .complex_value_layers import ComplexValueLSTMCell
from .complex_value_layers import ComplexValueDense


class ComplexVariables(object):
  """
  Complex Value NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # CCNN
    self.conv2d_layers = []
    if PARAM.no_cnn:
      pass
    else:
      conv2d_1 = ComplexValueConv2d(8, [5,5], padding="same", name='se_net/conv2_1') # -> [batch, time, fft_dot, 8]
      conv2d_2 = ComplexValueConv2d(16, [5,5], dilation_rate=[2,2], padding="same", name='se_net/conv2_2') # -> [batch, t, f, 16]
      conv2d_3 = ComplexValueConv2d(8, [5,5], dilation_rate=[4,4], padding="same", name='se_net/conv2_3') # -> [batch, t, f, 8]
      conv2d_4 = ComplexValueConv2d(1, [5,5], padding="same", name='se_net/conv2_4') # -> [batch, t, f, 1]
      self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]

    # CBLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      complex_lstm_cell_f = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.clstmCell_implementation)
      complex_lstm_cell_b = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.clstmCell_implementation)
      forward_lstm = tf.keras.layers.RNN(complex_lstm_cell_f, return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.RNN(complex_lstm_cell_b, return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='se_net/blstm_%d' % i)
      self.blstm_layers.append(blstm)

    # CLSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      complex_lstm_cell = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                               implementation=PARAM.clstmCell_implementation)
      lstm = tf.keras.layers.RNN(complex_lstm_cell, return_sequences=True, name='se_net/lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # CFC
    self.out_fc = ComplexValueDense(PARAM.fft_dot, name='se_net/out_fc')


class RealVariables(object):
  """
  Real Value NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # linear_coef and log_bias in log features # f = a*[log(bx+c)-log(c)], (a,b,c>0), init:a=0.1,b=1.0,c=1e-6
    self._f_log_a_var = tf.compat.v1.get_variable('LogFilter/f_log_a', dtype=tf.float32, # belong to discriminator
                                                  initializer=tf.constant(PARAM.f_log_a), trainable=PARAM.f_log_var_trainable)
    self._f_log_b_var = tf.compat.v1.get_variable('LogFilter/f_log_b', dtype=tf.float32,
                                                  initializer=tf.constant(PARAM.f_log_b), trainable=PARAM.f_log_var_trainable)
    self._f_log_c_var = tf.compat.v1.get_variable('LogFilter/f_log_c', dtype=tf.float32,
                                                  initializer=tf.constant(PARAM.f_log_c), trainable=False)
    self._f_log_a = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_a_var)
    self._f_log_b = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_b_var)
    self._f_log_c = PARAM.log_filter_eps_c + tf.nn.relu(self._f_log_c_var)

    # CNN
    self.conv2d_layers = []
    if PARAM.no_cnn:
      pass
    else:
      conv2d_1 = tf.keras.layers.Conv2D(16, [5,5], padding="same", name='se_net/conv2_1') # -> [batch, time, fft_dot, 8]
      conv2d_2 = tf.keras.layers.Conv2D(32, [5,5], dilation_rate=[2,2], padding="same", name='se_net/conv2_2') # -> [batch, t, f, 16]
      conv2d_3 = tf.keras.layers.Conv2D(16, [5,5], dilation_rate=[4,4], padding="same", name='se_net/conv2_3') # -> [batch, t, f, 8]
      conv2d_4 = tf.keras.layers.Conv2D(1, [5,5], padding="same", name='se_net/conv2_4') # -> [batch, t, f, 1]
      self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]

    # BLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2,
                                          implementation=PARAM.rlstmCell_implementation,
                                          return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2,
                                           implementation=PARAM.rlstmCell_implementation,
                                           return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='se_net/blstm_%d' % i)
      self.blstm_layers.append(blstm)
    # self.blstm_layers = []
    # if PARAM.blstm_layers > 0:
    #   forward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="fwlstm")
    #   backward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="bwlstm", go_backwards=True)
    #   self.blstm_layers.append(tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
    #                                                          merge_mode='concat', name='blstm'))

    #LSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                  return_sequences=True, implementation=PARAM.rlstmCell_implementation,
                                  name='se_net/lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.fft_dot, name='se_net/out_fc')

    # discriminator
    if PARAM.model_name == "DISCRIMINATOR_AD_MODEL":
      if PARAM.simple_D:
        self.d_blstm = tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_1')
        self.d_lstm = tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_2')
        self.d_denses = [tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_3'),
                         tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_4'),
                         tf.keras.layers.Dense(2, name='discriminator/d_dense_5')]
      else:
        forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=2,
                                            return_sequences=True, name='fwlstm_%d' % i)
        backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=2,
                                             return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
        self.d_blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                                     merge_mode='concat', name='discriminator/d_blstm')
        self.d_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=2,
                                           return_sequences=PARAM.frame_level_D, name='discriminator/lstm')
        self.d_denses = [tf.keras.layers.Dense(self.N_RNN_CELL//2, activation='relu', name='discriminator/d_dense_1'),
                         tf.keras.layers.Dense(2, name='discriminator/d_dense_2')]


class RCHybirdVariables(RealVariables):
  """
  Real complex hybird model  Variables
  """
  def __init__(self):
    super(RCHybirdVariables, self).__init__()
    # post complex net
    self.post_complex_layers = []
    for i in range(1, PARAM.post_lstm_layers+1):
      complex_lstm_cell_f = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                                 implementation=PARAM.clstmCell_implementation)
      fwlstm = tf.keras.layers.RNN(complex_lstm_cell_f, return_sequences=True, name='fw_complex_lstm_%d' % i)
      complex_lstm_cell_b = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                                 implementation=PARAM.clstmCell_implementation)
      bwlstm = tf.keras.layers.RNN(complex_lstm_cell_b, return_sequences=True, go_backwards=True, name='bw_complex_lstm_%d' % i)
      blstm = tf.keras.layers.Bidirectional(layer=fwlstm, backward_layer=bwlstm,
                                            merge_mode='concat', name='se_net/post_complex_blstm')
      self.post_complex_layers.append(blstm)

    if len(self.post_complex_layers) > 0:
      self.post_complex_layers.append(ComplexValueDense(PARAM.fft_dot, name='se_net/post_out_cfc'))


class RRHybirdVariables(RealVariables):
  """
  Real Real-post hybird model  Variables
  """
  def __init__(self):
    super(RRHybirdVariables, self).__init__()
    # post complex net
    self.post_real_layers = []
    for i in range(1, PARAM.post_lstm_layers+1):
      fwlstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                    implementation=PARAM.clstmCell_implementation,
                                    return_sequences=True, name='fw_real_lstm_%d' % i)
      bwlstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                    implementation=PARAM.clstmCell_implementation, go_backwards=True,
                                    return_sequences=True, name='bw_real_lstm_%d' % i)
      blstm = tf.keras.layers.Bidirectional(layer=fwlstm, backward_layer=bwlstm,
                                            merge_mode='concat', name='se_net/post_real_blstm')
      self.post_real_layers.append(blstm)

    if len(self.post_real_layers) > 0:
      self.post_real_layers.append(tf.keras.layers.Dense(PARAM.fft_dot*2, name='se_net/post_out_fc'))

class Module(object):
  """
  speech enhancement base.
  Discriminate spec and mag:
    spec: spectrum, complex value.
    mag: magnitude, real value.
  """
  def __init__(self,
               mode,
               variables: Union[RealVariables, ComplexVariables],
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    del noise_wav_batch
    self.mixed_wav_batch = mixed_wav_batch

    self.variables = variables
    self.mode = mode

    # global_step, lr, vars
    self._global_step = self.variables._global_step
    self._lr = self.variables._lr
    self.save_variables = [self.global_step, self._lr]

    # for lr halving
    self.new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self.assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for reset global_step
    self.new_step = tf.compat.v1.placeholder(tf.int32, name='new_step')
    self.assign_step = tf.compat.v1.assign(self._global_step, self.new_step)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)

    # nn forward
    forward_outputs = self.forward(mixed_wav_batch)
    self._est_clean_wav_batch = forward_outputs[-1]

    # labels
    if mode == PARAM.MODEL_INFER_KEY:
      clean_wav_batch = tf.ones_like(mixed_wav_batch)
    self.clean_wav_batch = clean_wav_batch
    self.clean_spec_batch = misc_utils.tf_batch_stft(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # complex label
    # self.noise_wav_batch = mixed_wav_batch - clean_wav_batch
    # self.noise_spec_batch = misc_utils.tf_batch_stft(self.noise_wav_batch, PARAM.frame_length, PARAM.frame_step)
    # self.nosie_mag_batch = tf.math.abs(self.noise_spec_batch)
    if PARAM.use_wav_as_feature:
      self.clean_mag_batch = self.clean_spec_batch
    elif PARAM.feature_type == "DFT":
      self.clean_mag_batch = tf.math.abs(self.clean_spec_batch) # mag label
    elif PARAM.feature_type == "DCT":
      self.clean_mag_batch = self.clean_spec_batch # DCT real feat

    self._se_loss = self.get_loss(forward_outputs)

    self._d_loss = tf.reduce_sum(tf.zeros([1]))
    self._deep_features_loss = 0.0
    self._deep_features_losses = 0.0
    if PARAM.model_name == "DISCRIMINATOR_AD_MODEL" and mode != PARAM.MODEL_INFER_KEY:
      self._d_loss, self._deep_features_losses = self.get_discriminator_loss(forward_outputs)
      for l in self._deep_features_losses:
        self._deep_features_loss += l

    self._loss = self._se_loss + self._d_loss

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.d_params = tf.compat.v1.trainable_variables(scope='discriminator*')
    if PARAM.add_logFilter_in_Discrimitor:
      self.d_params.extend(tf.compat.v1.trainable_variables(scope='LogFilter*'))
      # misc_utils.show_variables(d_params)
    self.se_net_params = tf.compat.v1.trainable_variables(scope='se_net*')
    self.save_variables.extend(self.se_net_params + self.d_params)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=PARAM.max_keep_ckpt,
                                          save_relative_paths=True)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    # optimizer
    # opt = tf.keras.optimizers.Adam(learning_rate=self._lr)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(self._lr)
    # misc_utils.show_variables(se_net_params)
    gradients = tf.gradients(
      self._se_loss,
      self.se_net_params,
      colocate_gradients_with_ops=True
    )
    self.se_loss_grads, gradient_norm = tf.clip_by_global_norm(
        gradients, PARAM.max_gradient_norm)
    self._train_op = self.optimizer.apply_gradients(zip(self.se_loss_grads, self.se_net_params),
                                                    global_step=self.global_step)


  def CNN_RNN_FC(self, mixed_mag_batch, training=False):
    outputs = tf.expand_dims(mixed_mag_batch, -1) # [batch, time, fft_dot, 1]
    _batch_size = tf.shape(outputs)[0]

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, fft_dot]
      if PARAM.cnn_shortcut == "add":
        outputs = tf.add(outputs, mixed_mag_batch)
      elif PARAM.cnn_shortcut == "multiply":
        outputs = tf.multiply(outputs, mixed_mag_batch)


    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # BLSTM
    self.blstm_outputs = []
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)
      self.blstm_outputs.append(outputs)

    # LSTM
    for lstm in self.variables.lstm_layers:
      outputs = lstm(outputs, training=training)

    # FC
    if len(self.variables.blstm_layers) > 0 and len(self.variables.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def real_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    if PARAM.use_wav_as_feature:
      mixed_mag_batch = mixed_spec_batch
    elif PARAM.feature_type == "DFT":
      mixed_mag_batch = tf.math.abs(mixed_spec_batch)
      self.mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    elif PARAM.feature_type == "DCT":
      mixed_mag_batch = mixed_spec_batch
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)

    if PARAM.add_logFilter_in_SE_inputs:
      a = tf.stop_gradient(self.variables._f_log_a)
      b = tf.stop_gradient(self.variables._f_log_b)
      c = tf.stop_gradient(self.variables._f_log_c)
      mixed_mag_batch = misc_utils.LogFilter_of_Loss(a, b, c, mixed_mag_batch,
                                                     PARAM.LogFilter_type)

    mask = self.CNN_RNN_FC(mixed_mag_batch, training)

    if PARAM.net_out_mask:
      est_clean_mag_batch = tf.multiply(mask, mixed_mag_batch) # mag estimated
    else:
      est_clean_mag_batch = mask

    if PARAM.feature_type == "DFT":
      est_clean_mag_batch = tf.nn.relu(est_clean_mag_batch)

    if PARAM.use_wav_as_feature:
      est_clean_spec_batch = est_clean_mag_batch
    elif PARAM.feature_type == "DFT":
      est_clean_spec_batch = tf.complex(est_clean_mag_batch, 0.0) * tf.exp(tf.complex(0.0, self.mixed_angle_batch)) # complex
    elif PARAM.feature_type == "DCT":
      est_clean_spec_batch = est_clean_mag_batch
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def post_real_networks_forward(self, input_mag_batch,  mixed_angle_batch, mixed_wav_len):
    assert PARAM.post_lstm_layers > 0, 'hybired rr model require PARAM.post_lstm_layers > 0 to use complex value post net'
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)

    if PARAM.complex_clip_mag is True:
      input_mag_batch = tf.clip_by_value(input_mag_batch, 0.0, float(PARAM.complex_clip_mag_max))
    input_spec_batch = tf.complex(input_mag_batch, 0.0) * tf.exp(tf.complex(0.0, mixed_angle_batch))

    outputs = tf.concat([tf.math.real(input_spec_batch), tf.math.imag(input_spec_batch)],
                        axis=-1)
    _batch_size = tf.shape(outputs)[0]
    # print(outputs.shape.as_list(), 'cnn_shape')

    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot*2])

    # CLSTM
    for blstm in self.variables.post_real_layers[:-1]:
      outputs = blstm(outputs, training=training)

    # CFC
    outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    outputs = self.variables.post_real_layers[-1](outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot*2])
    outputs_real, outputs_imag = tf.split(outputs, 2, axis=-1)
    outputs = tf.complex(outputs_real, outputs_imag)

    if PARAM.post_complex_net_output == 'cmask': # cmask, cresidual, cspec
      est_clean_spec_batch = tf.multiply(outputs, input_spec_batch)
    elif PARAM.post_complex_net_output == 'cresidual':
      est_clean_spec_batch = tf.add(outputs, input_spec_batch)
    elif PARAM.post_complex_net_output == 'cspec':
      est_clean_spec_batch = outputs

    _mixed_wav_len = mixed_wav_len
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    est_clean_mag_batch = tf.math.abs(est_clean_spec_batch)

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def post_complex_networks_forward(self, input_mag_batch, mixed_angle_batch, mixed_wav_len):

    assert PARAM.post_lstm_layers > 0, 'hybired rc model require PARAM.post_lstm_layers > 0 to use complex value post net'
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)

    if PARAM.complex_clip_mag is True:
      input_mag_batch = tf.clip_by_value(input_mag_batch, 0.0, float(PARAM.complex_clip_mag_max))
    input_spec_batch = tf.complex(input_mag_batch, 0.0) * tf.exp(tf.complex(0.0, mixed_angle_batch))
    outputs = input_spec_batch
    _batch_size = tf.shape(outputs)[0]
    # print(outputs.shape.as_list(), 'cnn_shape')

    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # CLSTM
    for blstm in self.variables.post_complex_layers[:-1]:
      outputs = blstm(outputs, training=training)

    # CFC
    outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    outputs = self.variables.post_complex_layers[-1](outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    if PARAM.post_complex_net_output == 'cmask': # cmask, cresidual, cspec
      est_clean_spec_batch = tf.multiply(outputs, input_spec_batch)
    elif PARAM.post_complex_net_output == 'cresidual':
      est_clean_spec_batch = tf.add(outputs, input_spec_batch)
    elif PARAM.post_complex_net_output == 'cspec':
      est_clean_spec_batch = outputs

    _mixed_wav_len = mixed_wav_len
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    est_clean_mag_batch = tf.math.abs(est_clean_spec_batch)

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def clean_and_enhanced_mag_discriminator(self, clean_mag_batch, est_mag_batch):
    deep_features = []
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    outputs = tf.concat([clean_mag_batch, est_mag_batch], axis=0)
    if PARAM.add_logFilter_in_Discrimitor:
      a = self.variables._f_log_a
      b = self.variables._f_log_b
      c = self.variables._f_log_c
      outputs = misc_utils.LogFilter_of_Loss(a,b,c,outputs,PARAM.LogFilter_type)

    # deep_features.append(outputs) # [batch*2, time, f]
    if PARAM.frame_level_D or PARAM.simple_D:
      zeros = tf.zeros(clean_mag_batch.shape[0:2], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0:2], dtype=tf.int32)
    else:
      zeros = tf.zeros(clean_mag_batch.shape[0], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0], dtype=tf.int32)
    labels = tf.concat([zeros, ones], axis=0)
    onehot_labels = tf.one_hot(labels, 2)
    # print(outputs.shape.as_list(), ' dddddddddddddddddddddd test shape')

    if PARAM.simple_D:
      outputs1 = self.variables.d_blstm(outputs) # [batch*2, time, fea]
      deep_features.append(outputs1) # [batch*2 time f]

      outputs2 = self.variables.d_lstm(outputs1) # [batch*2, time, f]
      deep_features.append(outputs2)

      outputs3 = self.variables.d_denses[0](outputs2)
      deep_features.append(outputs3)

      # inputs4 = tf.concat([outputs3, outputs2], axis=-1)
      inputs4 = outputs3
      outputs4 = self.variables.d_denses[1](inputs4)
      deep_features.append(outputs4)

      # inputs5 = tf.concat([outputs4, outputs1], axis=-1)
      inputs5 = outputs4
      logits = self.variables.d_denses[2](inputs5)
    else:
      outputs = self.variables.d_blstm(outputs, training=training) # [batch*2, time, fea]
      deep_features.append(outputs) # [batch*2 time f]
      outputs = self.variables.d_lstm(outputs, training=training) # [batch, fea] or [batch*2, time, f]
      deep_features.append(outputs)
      for dense in self.variables.d_denses:
        outputs = dense(outputs)
        deep_features.append(outputs)
      logits = outputs # [batch*2, time, 2]
    return logits, onehot_labels, deep_features


  def CCNN_CRNN_CFC(self, mixed_spec_batch, training=False):
    mixed_spec_batch = tf.expand_dims(mixed_spec_batch, -1) # [batch, time, fft_dot, 1]
    outputs = mixed_spec_batch
    _batch_size = tf.shape(outputs)[0]
    # print(outputs.shape.as_list(), 'cnn_shape')

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
      # print(outputs.shape.as_list(), 'cnn_shape')
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, fft_dot]

    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # CBLSTM
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)

    # CLSTM
    for lstm in self.variables.lstm_layers:
      outputs = lstm(outputs, training=training)

    # CFC
    if len(self.variables.blstm_layers) > 0 and len(self.variables.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def complex_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    # clip mag
    if PARAM.complex_clip_mag is True:
      mixed_mag_batch = tf.abs(mixed_spec_batch)
      # self.debug_mag = mixed_mag_batch
      mixed_angle_batch = tf.angle(mixed_spec_batch)
      mixed_mag_batch = tf.clip_by_value(mixed_mag_batch, 0.0, float(PARAM.complex_clip_mag_max))
      mixed_spec_batch = tf.complex(mixed_mag_batch, 0.0) * tf.exp(tf.complex(0.0, mixed_angle_batch))

    complex_mask = self.CCNN_CRNN_CFC(mixed_spec_batch, training)
    if PARAM.net_out_mask:
      est_clean_spec_batch = c_ops.tf_complex_multiply(complex_mask, mixed_spec_batch) # mag estimated
    else:
      est_clean_spec_batch = complex_mask
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    est_clean_mag_batch = tf.math.abs(est_clean_spec_batch)

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  @abc.abstractmethod
  def forward(self, mixed_wav_batch):
    """
    Returns:
      forward_outputs: pass to get_loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "forward not implement, code: 939iddfoollvoae")


  @abc.abstractmethod
  def get_loss(self, forward_outputs):
    """
    Returns:
      loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "get_loss not implement, code: 67hjrethfd")


  @abc.abstractmethod
  def get_discriminator_loss(self, forward_outputs):
    """
    Returns:
      loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "get_discriminator_loss not implement, code: qyhhtwgrff")


  def change_lr(self, sess, new_lr):
    sess.run(self.assign_lr, feed_dict={self.new_lr:new_lr})

  def change_global_step(self, sess, new_step):
    sess.run(self.assign_step, feed_dict={self.new_step:new_step})

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_op(self):
    return self._train_op

  @property
  def loss(self):
    return self._loss

  @property
  def d_loss(self):
    return self._d_loss

  @property
  def lr(self):
    return self._lr

  @property
  def est_clean_wav_batch(self):
    return self._est_clean_wav_batch
