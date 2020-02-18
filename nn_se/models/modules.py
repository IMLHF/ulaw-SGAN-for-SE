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
      conv2d_1 = tf.keras.layers.Conv2D(16, [5,5], padding="same", name='se_net/conv2_1') # -> [batch, time, feature_dim, 8]
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
    self.out_fc = tf.keras.layers.Dense(PARAM.feature_dim, name='se_net/out_fc')

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

    self.D_out_fc = tf.keras.layers.Dense(3 if PARAM.add_noisy_class_in_D else 2, name='discriminator/out_fc')

    ### FeatureTransformerLayers
    if "LogValueT" in PARAM.FT_type:
      ## 1. linear_coef and log_bias in log features
      self._f_log_a_var = tf.compat.v1.get_variable('FeatureTransformerLayer/f_log_a', dtype=tf.float32, # belong to discriminator
                                                    initializer=tf.constant(PARAM.f_log_a), trainable=PARAM.f_log_var_trainable)
      self._f_log_b_var = tf.compat.v1.get_variable('FeatureTransformerLayer/f_log_b', dtype=tf.float32,
                                                    initializer=tf.constant(PARAM.f_log_b), trainable=PARAM.f_log_var_trainable)
      self._f_log_a = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_a_var)
      self._f_log_b = PARAM.log_filter_eps_a_b + tf.nn.relu(self._f_log_b_var)

      def LogFilter_of_Loss(x,type_=PARAM.LogFilter_type):
        a = self._f_log_a
        b = self._f_log_b
        b_times = PARAM.logFT_type2_btimes
        if type_ == 0:
          raise NotImplementedError
        elif type_ == 1: # u-low transformer
          y = tf.log(x * b + 1.0 + 0.0*a) / tf.log(b + 1.0)
        elif type_ == 2: # u-low transformer add speed up b changing
          y = tf.log(x * b_times * b + 1.0 + 0.0*a) / tf.log(b_times * b + 1.0)
        elif type_ == 3: # only B
          y = (tf.log(x * b + 0.001 + 0.0*a) - tf.log(0.001)) / (tf.log(b + 0.001)-tf.log(0.001))
        return y
      self.LogFilterT = LogFilter_of_Loss

    if "RandomDenseT" in PARAM.FT_type:
      ## 2. RandomDenseT
      self.RandomDenseT = tf.keras.layers.Dense(self.N_RNN_CELL, activation='tanh',
                                                name='FeatureTransformerLayer/FT_Dense')
    if "MelDenseT" in PARAM.FT_type:
      ## 3. MelDenseT
      melmat_fun = tf.contrib.signal.linear_to_mel_weight_matrix
      melMatrix = tf.compat.v1.get_variable('FeatureTransformerLayer/FT_MelMat', dtype=tf.float32,
                                            trainable=PARAM.melDenseT_trainable,
                                            initializer=melmat_fun(PARAM.MelDenseT_n_mel, PARAM.feature_dim,
                                                                   PARAM.sampling_rate, 0, PARAM.sampling_rate//2))
      self.melMatrix = melMatrix

      def stft2mel(x):
        return tf.matmul(x, melMatrix)
      self.MelDenseT = stft2mel

    def FeatureTransformer(x):
      for ft_type in PARAM.FT_type:
        if ft_type == "LogValueT":
          x = self.LogFilterT(x)
        elif ft_type == "RandomDenseT":
          x = self.RandomDenseT(x)
        elif ft_type == "MelDenseT":
          assert PARAM.feature_type == "AbsDFT", 'Feature type is not AbsDFT, so cannot apple MelDenseT'
          x = self.MelDenseT(x)
        else:
          raise NotImplementedError
      return x
    self.FeatureTransformer = FeatureTransformer


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
               noise_wav_batch=None):
    del noise_wav_batch
    self.mixed_wav_batch = mixed_wav_batch # for placeholder

    # save mixed angle
    self.mixed_angle_batch = tf.angle(misc_utils.tf_wav2DFT(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step))

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
    forward_outputs = self.forward(self.mixed_wav_batch)
    self._est_clean_wav_batch = forward_outputs[-1]


    # get label and loss
    if mode != PARAM.MODEL_INFER_KEY:
      # labels
      self.clean_wav_batch = clean_wav_batch
      self.clean_spec_batch = misc_utils.tf_wav2DFT(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # complex label
      self.clean_mag_batch = misc_utils.tf_wav2AbsDFT(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # mag label
      # self.noise_wav_batch = mixed_wav_batch - clean_wav_batch
      # self.noise_spec_batch = misc_utils.tf_wav2feature(self.noise_wav_batch, PARAM.frame_length, PARAM.frame_step)
      # self.nosie_mag_batch = tf.math.abs(self.noise_spec_batch)

      self.mixed_mag_batch = misc_utils.tf_wav2AbsDFT(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step) # as D inputs

      # losses
      self._not_transformed_losses, self._transformed_losses, self._d_loss = self.get_loss(forward_outputs)

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.d_params = tf.compat.v1.trainable_variables(scope='discriminator/')
    self.ft_params = tf.compat.v1.trainable_variables(scope='FeatureTransformerLayer/')
    self.se_net_params = tf.compat.v1.trainable_variables(scope='se_net/')

    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nD PARAMs:")
      misc_utils.show_variables(self.d_params)
      print("\nft PARAMs")
      misc_utils.show_variables(self.ft_params)
      print("\nSE PARAMs")
      misc_utils.show_variables(self.se_net_params)

    self.save_variables.extend(self.se_net_params + self.d_params + self.ft_params)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=PARAM.max_keep_ckpt,
                                          save_relative_paths=True)

    if mode == PARAM.MODEL_VALIDATE_KEY:
      return
    if mode == PARAM.MODEL_INFER_KEY:
      return

  def CNN_RNN_FC(self, input_feature_batch, training=False):
    if PARAM.add_FeatureTrans_in_SE_inputs:
      input_feature_batch = self.variables.FeatureTransformer(input_feature_batch)
    outputs = tf.expand_dims(input_feature_batch, -1) # [batch, time, feature_dim, 1]
    _batch_size = tf.shape(outputs)[0]

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, feature_dim]
      if PARAM.cnn_shortcut == "add":
        outputs = tf.add(outputs, input_feature_batch)
      elif PARAM.cnn_shortcut == "multiply":
        outputs = tf.multiply(outputs, input_feature_batch)
      else: #None
        pass


    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.feature_dim])

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
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.feature_dim])
    return outputs


  def real_networks_forward(self, mixed_wav_batch):
    input_feature_batch = misc_utils.tf_wav2feature(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)

    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    nn_out = self.CNN_RNN_FC(input_feature_batch, training)

    if PARAM.net_out_mask:
      if PARAM.mask_type == "IRM":
        est_feature_batch = tf.multiply(nn_out, input_feature_batch) # mag estimated
      elif PARAM.mask_type == "cIRM":
        mask_real, mask_imag = tf.split(nn_out, 2, axis=-1)
        cIRM = tf.complex(mask_real, mask_imag)
        mixed_cstft_real, mixed_cstft_imag = tf.split(input_feature_batch, 2, axis=-1)
        mixed_cstft = tf.complex(mixed_cstft_real, mixed_cstft_imag)
        est_stft = tf.multiply(cIRM, mixed_cstft)
        est_real = tf.real(est_stft)
        est_imag = tf.imag(est_stft)
        est_feature_batch = tf.concat([est_real, est_imag], axis=-1)
    else:
      est_feature_batch = nn_out

    if PARAM.feature_type == "WAV":
      est_clean_wav_batch = misc_utils.tf_feature2wav(est_feature_batch, PARAM.frame_length, PARAM.frame_step)
      est_clean_mag_batch = misc_utils.tf_wav2AbsDFT(est_clean_wav_batch)
      est_clean_spec_batch = misc_utils.tf_wav2DFT(est_clean_wav_batch)
    elif PARAM.feature_type == "AbsDFT":
      est_clean_mag_batch = tf.nn.relu(est_feature_batch)
      est_clean_spec_batch = tf.complex(est_feature_batch, 0.0) * tf.exp(tf.complex(0.0, self.mixed_angle_batch))
      est_clean_wav_batch = misc_utils.tf_feature2wav((est_clean_mag_batch, self.mixed_angle_batch),
                                                      PARAM.frame_length, PARAM.frame_step)
    elif PARAM.feature_type == "DCT":
      est_clean_mag_batch = est_feature_batch
      est_clean_spec_batch = est_feature_batch
      est_clean_wav_batch = misc_utils.tf_feature2wav(est_feature_batch, PARAM.frame_length, PARAM.frame_step)
    elif PARAM.feature_type == "ComplexDFT":
      est_clean_wav_batch = misc_utils.tf_feature2wav(est_feature_batch, PARAM.frame_length, PARAM.frame_step)
      cstft_real, cstft_imag = tf.split(est_feature_batch, 2, axis=-1)
      est_clean_spec_batch = tf.complex(cstft_real, cstft_imag)
      est_clean_mag_batch = tf.abs(est_clean_spec_batch)

    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    est_clean_wav_batch = tf.slice(est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def clean_and_enhanced_mag_discriminator(self, clean_mag_batch, est_mag_batch, mixed_mag_batch):
    deep_features = []
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    if PARAM.add_noisy_class_in_D:
      outputs = tf.concat([clean_mag_batch, est_mag_batch, mixed_mag_batch], axis=0)
    else:
      outputs = tf.concat([clean_mag_batch, est_mag_batch], axis=0)

    # Features Transformer
    outputs = self.variables.FeatureTransformer(outputs)
    _batch_size = outputs.shape[0]

    # deep_features.append(outputs) # [batch*2, time, f]
    if PARAM.frame_level_D:
      zeros = tf.zeros(clean_mag_batch.shape[0:2], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0:2], dtype=tf.int32)
      twos = tf.ones(mixed_mag_batch.shape[0:2], dtype=tf.int32) * 2
    else:
      zeros = tf.zeros(clean_mag_batch.shape[0], dtype=tf.int32)
      ones = tf.ones(est_mag_batch.shape[0], dtype=tf.int32)
      twos = tf.ones(mixed_mag_batch.shape[0], dtype=tf.int32) * 2
    if PARAM.add_noisy_class_in_D:
      labels = tf.concat([zeros, ones, twos], axis=0)
    else:
      labels = tf.concat([zeros, ones], axis=0)
    onehot_labels = tf.one_hot(labels, 3 if PARAM.add_noisy_class_in_D else 2)
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
      logits = tf.reshape(outputs, [_batch_size, -1, 3 if PARAM.add_noisy_class_in_D else 2])
    return logits, onehot_labels, deep_features


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
