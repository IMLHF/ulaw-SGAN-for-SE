import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils


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

    #LSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                  return_sequences=True, implementation=PARAM.rlstmCell_implementation,
                                  name='se_net/lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.fft_dot, name='se_net/out_fc')

    #### discriminator
    # self.d_denses = [tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_1'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_2'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_3'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_4'),
    #                  tf.keras.layers.Dense(2, name='discriminator/d_dense_5')]
    self.D_blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2,
                                          implementation=PARAM.rlstmCell_implementation,
                                          return_sequences=True, name='dfwlstm_%d' % i)
      backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2,
                                           implementation=PARAM.rlstmCell_implementation,
                                           return_sequences=True, name='dbwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='discriminator/blstm_%d' % i)
      self.D_blstm_layers.append(blstm)

    self.D_lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                  return_sequences=True, implementation=PARAM.rlstmCell_implementation,
                                  name='discriminator/lstm_%d' % i)
      self.Dlstm_layers.append(lstm)

    self.D_out_fc = tf.keras.layers.Dense(2, name='discriminator/out_fc')

    # FeatureTransformerLayer
    if PARAM.FT_type == "LogValueT":
      # linear_coef and log_bias in log features
      self._f_log_a_var = tf.compat.v1.get_variable('FeatureTransformerLayer/f_log_a', dtype=tf.float32, # belong to discriminator
                                                    initializer=tf.constant(PARAM.f_log_a), trainable=PARAM.f_log_var_trainable)
      self._f_log_b_var = tf.compat.v1.get_variable('FeatureTransformerLayer/f_log_b', dtype=tf.float32,
                                                    initializer=tf.constant(PARAM.f_log_b), trainable=PARAM.f_log_var_trainable)
      self._f_log_a = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_a_var)
      self._f_log_b = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_b_var)

      def LogFilter_of_Loss(x,type_=PARAM.LogFilter_type):
        a = self._f_log_a
        b = self._f_log_b
        if type_ == 0:
          raise NotImplementedError
        elif type_ == 1: # u-low transformer
          y = tf.log(x * b + 1.0 + 0.0*a) / tf.log(b + 1.0)
        elif type_ ==2: # modified u-low transformer
          y = tf.log(x * b + 1.0) / tf.log(a * b + 1.0)
        return y
      self.FeatureTransformer = LogFilter_of_Loss
    elif PARAM.FT_type == "DenseT":
      self.FeatureTransformer = tf.keras.layers.Dense(self.N_RNN_CELL, activation='tanh',
                                                      name='FeatureTransformerLayer/FT_Dense')
    else:
      raise NotImplementedError


class Module(object):
  """
  speech enhancement base.
  Discriminate spec and mag:
    spec: spectrum, complex value.
    mag: magnitude, real value.
  """
  def __init__(self,
               mode,
               variables: Union[RealVariables],
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None,
               noisy_wav_batch_noLabel=None):
    del noise_wav_batch
    self.mixed_wav_batch = mixed_wav_batch # for placeholder
    mixed_wav_batch_extend = mixed_wav_batch
    if PARAM.use_noLabel_noisy_speech:
      self.label_noisy_batchsize = tf.shape(mixed_wav_batch)[0]
      self.noLabel_noisy_batchsize = tf.shape(noisy_wav_batch_noLabel)[0]
      mixed_wav_batch_extend = tf.concat([mixed_wav_batch, noisy_wav_batch_noLabel], axis=0)
    else:
      del noisy_wav_batch_noLabel

    self.variables = variables
    self.mode = mode

    # global_step, lr, vars
    self._global_step = self.variables._global_step
    self._lr = self.variables._lr
    self.save_variables = [self.global_step, self._lr]

    # for lr halving
    self.new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self.assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)

    # nn forward
    forward_outputs = self.forward(mixed_wav_batch_extend)
    self._est_clean_wav_batch = forward_outputs[-1]

    # get label and loss
    if mode != PARAM.MODEL_INFER_KEY:
      # labels
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

      # losses
      self._not_transformed_losses, self._transformed_losses, self._d_loss = self.get_loss(forward_outputs)

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.d_params = tf.compat.v1.trainable_variables(scope='discriminator*')
    self.d_params.extend(tf.compat.v1.trainable_variables(scope='FeatureTransformerLayer*'))
    # misc_utils.show_variables(d_params)
    self.se_net_params = tf.compat.v1.trainable_variables(scope='se_net*')
    self.save_variables.extend(self.se_net_params + self.d_params)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=PARAM.max_keep_ckpt,
                                          save_relative_paths=True)

    if mode == PARAM.MODEL_VALIDATE_KEY:
      return
    if mode == PARAM.MODEL_INFER_KEY:
      return

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
    # self.blstm_outputs = []
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)
      # self.blstm_outputs.append(outputs)

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


  def real_networks_forward(self, mixed_wav_batch_extend):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch_extend, PARAM.frame_length, PARAM.frame_step)
    if PARAM.use_wav_as_feature:
      mixed_mag_batch = mixed_spec_batch
    elif PARAM.feature_type == "DFT":
      mixed_mag_batch = tf.math.abs(mixed_spec_batch)
      self.mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    elif PARAM.feature_type == "DCT":
      mixed_mag_batch = mixed_spec_batch
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    input_feature = mixed_mag_batch

    if PARAM.add_FeatureTrans_in_SE_inputs:
      input_feature = self.variables.FeatureTransformer(mixed_mag_batch)

    mask = self.CNN_RNN_FC(input_feature, training)

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
    _mixed_wav_len = tf.shape(mixed_wav_batch_extend)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def clean_and_enhanced_mag_discriminator(self, clean_mag_batch, est_mag_batch):
    deep_features = []
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    outputs = tf.concat([clean_mag_batch, est_mag_batch], axis=0)

    # Features Transformer
    outputs = self.variables.FeatureTransformer(outputs)
    _batch_size = outputs.shape[0]

    # deep_features.append(outputs) # [batch*2, time, f]
    if PARAM.frame_level_D:
      zeros = tf.zeros(clean_mag_batch.shape[0:2], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0:2], dtype=tf.int32)
    else:
      zeros = tf.zeros(clean_mag_batch.shape[0], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0], dtype=tf.int32)
    labels = tf.concat([zeros, ones], axis=0)
    onehot_labels = tf.one_hot(labels, 2)
    # print(outputs.shape.as_list(), ' dddddddddddddddddddddd test shape')

    # outputs1 = self.variables.d_denses[0](outputs) # [batch*2, time, fea]
    # deep_features.append(outputs1) # [batch*2 time f]

    # outputs2 = self.variables.d_denses[1](outputs1) # [batch*2, time, f]
    # if training:
    #   outputs2 = tf.nn.dropout(outputs2, keep_prob=PARAM.D_keep_prob)
    # deep_features.append(outputs2)

    # outputs3 = self.variables.d_denses[2](outputs2)
    # deep_features.append(outputs3)

    # # inputs4 = tf.concat([outputs3, outputs2], axis=-1)
    # inputs4 = outputs3
    # outputs4 = self.variables.d_denses[3](inputs4)
    # if training:
    #   outputs4 = tf.nn.dropout(outputs4, keep_prob=PARAM.D_keep_prob)
    # deep_features.append(outputs4)

    # # inputs5 = tf.concat([outputs4, outputs1], axis=-1)
    # inputs5 = outputs4
    # logits = self.variables.d_denses[4](inputs5)

    for blstm in self.variables.D_blstm_layers:
      outputs = blstm(outputs, training=training)

    for lstm in self.variables.D_lstm_layers:
      outputs = lstm(outputs, training=training)

    if not PARAM.frame_level_D:
      outputs = outputs[:,-1,:]

    # FC
    if len(self.variables.D_blstm_layers) > 0 and len(self.variables.D_lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.D_out_fc(outputs)
    logits = outputs
    if PARAM.frame_level_D:
      logits = tf.reshape(outputs, [_batch_size, -1, 2])
    return logits, onehot_labels, deep_features


  @abc.abstractmethod
  def forward(self, mixed_wav_batch_extend):
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
      not_transformed_loss, transformed_loss, d_loss
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

  @property
  def global_step(self):
    return self._global_step

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch

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
