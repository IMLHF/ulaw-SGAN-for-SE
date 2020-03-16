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

    # BLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                         #  dropout=0.2,
                                                         #  implementation=PARAM.rlstmCell_implementation,
                                                         return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                          # dropout=0.2,
                                                          # implementation=PARAM.rlstmCell_implementation,
                                                          return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='G/blstm_%d' % i)
      dropL = tf.keras.layers.Dropout(rate=0.2)
      self.blstm_layers.append(blstm)
      self.blstm_layers.append(dropL)

    #LSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                 #  dropout=0.2, recurrent_dropout=0.1,
                                                 return_sequences=True,
                                                 implementation=PARAM.rlstmCell_implementation,
                                                 name='G/lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.feature_dim, name='G/out_fc')

    #### discriminator
    # self.d_denses = [tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_1'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_2'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL//2, name='discriminator/d_dense_3'),
    #                  tf.keras.layers.Dense(self.N_RNN_CELL, name='discriminator/d_dense_4'),
    #                  tf.keras.layers.Dense(2, name='discriminator/d_dense_5')]
    self.D_blstm_layers = []
    for i in range(1, PARAM.D_blstm_layers+1):
      # tf.keras.layers.LSTM
      forward_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                         #  dropout=0.2,
                                                         #  implementation=PARAM.rlstmCell_implementation,
                                                         return_sequences=True, name='dfwlstm_%d' % i)
      backward_lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                          # dropout=0.2,
                                                          # implementation=PARAM.rlstmCell_implementation,
                                                          return_sequences=True, name='dbwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='D/blstm_%d' % i)
      dropL = tf.keras.layers.Dropout(rate=0.2)
      self.D_blstm_layers.append(blstm)
      self.D_blstm_layers.append(dropL)

    self.D_lstm_layers = []
    for i in range(1, PARAM.D_lstm_layers+1):
      lstm = tf.compat.v1.keras.layers.CuDNNLSTM(self.N_RNN_CELL,
                                                 #  dropout=0.2,
                                                 #  recurrent_dropout=0.1,
                                                 return_sequences=True,
                                                 implementation=PARAM.rlstmCell_implementation,
                                                 name='D/lstm_%d' % i)
      self.Dlstm_layers.append(lstm)

    self.D_out_fc = tf.keras.layers.Dense(3 if PARAM.add_noisy_class_in_D else 2, name='D/out_fc')

    ### FeatureTransformerLayers
    if "trainableUlaw" in PARAM.FT_type:
      # belong to discriminator
      self._f_u_var = tf.compat.v1.get_variable('D/FTL/f_u', dtype=tf.float32,
                                                initializer=tf.constant(PARAM.f_u),
                                                trainable=PARAM.f_u_var_trainable)
      self._f_u = PARAM.u_eps + tf.abs(self._f_u_var)

      def ulaw_fn(x):
        # x: [batch, time, fea]
        u = self._f_u
        u_times = PARAM.u_times
        y = tf.log(x * u_times * u + 1.0) / tf.log(u_times * u + 1.0)
        return y
      self.ulaw_fn = ulaw_fn

    if "trainableUlaw_v2" in PARAM.FT_type:
      # belong to discriminator
      self._f_u_var = tf.compat.v1.get_variable('D/FTL/f_u', shape=[1024], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(stddev=0.01),
                                                trainable=PARAM.f_u_var_trainable)
      self._f_u = PARAM.u_eps + tf.abs(tf.reduce_sum(self._f_u_var))

      def ulaw_fnv2(x):
        # x: [batch, time, fea]
        u = self._f_u
        y = tf.log(x * u + 1.0) / tf.log(u + 1.0)
        return y
      self.ulaw_fnv2 = ulaw_fnv2
    # if "RandomDenseT" in PARAM.FT_type:
    #   ## 3. RandomDenseT
    #   self.RandomDenseT = tf.keras.layers.Dense(self.N_RNN_CELL, activation='tanh',
    #                                             name='D/FTL/FT_Dense')
    # if "MelDenseT" in PARAM.FT_type:
    #   ## 4. MelDenseT
    #   melmat_fun = tf.contrib.signal.linear_to_mel_weight_matrix
    #   melMatrix = tf.compat.v1.get_variable('D/FTL/FT_MelMat', dtype=tf.float32,
    #                                         trainable=PARAM.melDenseT_trainable,
    #                                         initializer=melmat_fun(PARAM.MelDenseT_n_mel, PARAM.feature_dim,
    #                                                                PARAM.sampling_rate, 0, PARAM.sampling_rate//2))
    #   self.melMatrix = melMatrix
    #   def stft2mel(x):
    #     return tf.matmul(x, melMatrix)
    #   self.MelDenseT = stft2mel


    def FeatureTransformer(x):
      for ft_type in PARAM.FT_type:
        if ft_type == "trainableUlaw":
          x = self.ulaw_fn(x)
        elif ft_type == "trainableUlaw_v2":
          x = self.ulaw_fnv2(x)
        # elif ft_type == "RandomDenseT":
        #   x = self.RandomDenseT(x)
        # elif ft_type == "MelDenseT":
        #   x = self.MelDenseT(x)
        else:
          raise NotImplementedError
      return x
    self.FeatureTransformer = FeatureTransformer

class WavFeatures(
    collections.namedtuple("WavFeatures",
                           ("wav_batch", # [N, L]
                            "stft_batch", #[N, T, F]
                            "mag_batch", # [N, T, F]
                            "angle_batch", # [N, T, F]
                            "normed_stft_batch", # [N, T, F]
                            ))):
  pass

class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss_G", "sum_loss_D",
                            "show_losses", "stop_criterion_loss"))):
  pass

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
               clean_wav_batch=None):
    self.mode = mode
    self.variables = variables
    mixed_stft_batch = misc_utils.tf_wav2stft(mixed_wav_batch, PARAM.frame_length,
                                              PARAM.frame_step, PARAM.fft_length)  # [N, T, F]
    mixed_mag_batch = tf.abs(mixed_stft_batch) # [N, F, T]
    mixed_angle_batch = tf.angle(mixed_stft_batch)
    mixed_normed_stft_batch = tf.exp(tf.complex(0.0, mixed_angle_batch))
    self.mixed_wav_features = WavFeatures(wav_batch=mixed_wav_batch,
                                          stft_batch=mixed_stft_batch,
                                          mag_batch=mixed_mag_batch,
                                          angle_batch=mixed_angle_batch,
                                          normed_stft_batch=mixed_normed_stft_batch)
    if mode != PARAM.MODEL_INFER_KEY:
      # get label and loss
      clean_stft_batch = misc_utils.tf_wav2stft(clean_wav_batch, PARAM.frame_length,
                                                PARAM.frame_step, PARAM.fft_length)
      clean_mag_batch = tf.abs(clean_stft_batch)
      clean_angle_batch = tf.angle(clean_stft_batch)
      clean_normed_stft_batch = tf.exp(tf.complex(0.0, clean_angle_batch))
      self.clean_wav_features = WavFeatures(wav_batch=clean_wav_batch,
                                            stft_batch=clean_stft_batch,
                                            mag_batch=clean_mag_batch,
                                            angle_batch=clean_angle_batch,
                                            normed_stft_batch=clean_normed_stft_batch)


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

    # nn_se forward
    self.est_wav_features = self._forward()

    if self.mode != PARAM.MODEL_INFER_KEY:
      self._d_logits, self._d_labels = None, None
      if len(PARAM.sum_losses_D):
        self._d_logits, self._d_labels = self._discriminator(self.clean_wav_features.mag_batch,
                                                             self.est_wav_features.mag_batch,
                                                             self.mixed_wav_features.mag_batch)

      # losses
      self._losses = self._get_losses(self.est_wav_features, self.clean_wav_features,
                                      self._d_logits, self._d_labels)

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.d_params = tf.compat.v1.trainable_variables(scope='D/')
    self.g_params = tf.compat.v1.trainable_variables(scope='G/')

    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nD PARAMs:")
      misc_utils.show_variables(self.d_params)
      print("\nG PARAMs")
      misc_utils.show_variables(self.g_params)

    self.save_variables.extend(self.g_params + self.d_params)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=40,
                                          save_relative_paths=True)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    all_grads = []
    all_params = []

    ## G grads
    grads_G = tf.gradients(
      self._losses.sum_loss_G,
      self.g_params,
      colocate_gradients_with_ops=True
    )
    grads_G, _ = tf.clip_by_global_norm(grads_G, PARAM.max_gradient_norm)
    all_grads.extend(grads_G)
    all_params.extend(self.g_params)

    if len(PARAM.sum_losses_D)>0:
      ## D grads
      grads_D = tf.gradients(
        self._losses.sum_loss_D,
        self.d_params,
        colocate_gradients_with_ops=True
      )
      grads_D, _ = tf.clip_by_global_norm(grads_D, PARAM.max_gradient_norm)
      all_grads.extend(grads_D)
      all_params.extend(self.d_params)

    if PARAM.optimizer == "Adam":
      self.optimizer = tf.compat.v1.train.AdamOptimizer(self._lr)
    elif PARAM.optimizer == "RMSProp":
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self._lr)
    self._train_op = self.optimizer.apply_gradients(zip(all_grads, all_params),
                                                    global_step=self.global_step)


  def _RNN_FC(self, input_feature_batch, training=False):
    outputs = input_feature_batch # [batch, time, feature_dim]
    _batch_size = tf.shape(outputs)[0]

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


  def _forward(self):
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    nn_out = self._RNN_FC(self.mixed_wav_features.mag_batch, training)

    if PARAM.net_out_mask:
      est_feature_batch = tf.multiply(nn_out, self.mixed_wav_features.mag_batch) # mag estimated
    else:
      est_feature_batch = nn_out

    est_mag_batch = tf.nn.relu(est_feature_batch)
    est_stft_batch = tf.complex(est_mag_batch, 0.0) * self.mixed_wav_features.normed_stft_batch
    est_wav_batch = misc_utils.tf_stft2wav(est_stft_batch, PARAM.frame_length,
                                           PARAM.frame_step, PARAM.fft_length)
    _mixed_wav_len = tf.shape(self.mixed_wav_features.wav_batch)[-1]
    est_wav_batch = tf.slice(est_wav_batch, [0,0], [-1, _mixed_wav_len])
    return WavFeatures(wav_batch=est_wav_batch,
                       mag_batch=est_mag_batch,
                       stft_batch=est_stft_batch,
                       normed_stft_batch=self.mixed_wav_features.normed_stft_batch,
                       angle_batch=self.mixed_wav_features.angle_batch)


  def _discriminator(self, clean_mag_batch, est_mag_batch, mixed_mag_batch):
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    clean_fea_batch = clean_mag_batch
    est_fea_batch = est_mag_batch
    mixed_fea_batch = mixed_mag_batch
    if PARAM.add_noisy_class_in_D:
      outputs = tf.concat([clean_fea_batch, est_fea_batch, mixed_fea_batch], axis=0)
    else:
      outputs = tf.concat([clean_fea_batch, est_fea_batch], axis=0)

    # Features Transformer
    outputs = self.variables.FeatureTransformer(outputs)
    _batch_size = outputs.shape[0]

    if PARAM.frame_level_D:
      zeros = tf.zeros(clean_fea_batch.shape[0:2], dtype=tf.int32)
      ones = tf.ones(est_fea_batch.shape[0:2], dtype=tf.int32)
      twos = tf.ones(mixed_fea_batch.shape[0:2], dtype=tf.int32) * 2
    else:
      zeros = tf.zeros(clean_fea_batch.shape[0], dtype=tf.int32)
      ones = tf.ones(est_fea_batch.shape[0], dtype=tf.int32)
      twos = tf.ones(mixed_fea_batch.shape[0], dtype=tf.int32) * 2
    if PARAM.add_noisy_class_in_D:
      labels = tf.concat([zeros, ones, twos], axis=0)
    else:
      labels = tf.concat([zeros, ones], axis=0)
    onehot_labels = tf.one_hot(labels, 3 if PARAM.add_noisy_class_in_D else 2)
    # print(outputs.shape.as_list(), ' dddddddddddddddddddddd test shape')

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
    return logits, onehot_labels

  def _get_losses(self, est_wav_features, clean_wav_features,
                  d_logits, d_labels):

    def FixULawT_fn(x, u):
      # x: [batch, time, fea]
      y = tf.log(x * u + 1.0) / tf.log(u + 1.0)
      return y

    est_mag_batch = est_wav_features.mag_batch
    est_stft_batch = est_wav_features.stft_batch
    est_wav_batch = est_wav_features.wav_batch
    est_normed_stft_batch = est_wav_features.normed_stft_batch
    est_wav_ulawWav_batch = FixULawT_fn(est_wav_batch, 255.0)

    clean_mag_batch = clean_wav_features.mag_batch
    clean_stft_batch = clean_wav_features.stft_batch
    clean_wav_batch = clean_wav_features.wav_batch
    clean_wav_ulawWav_batch = FixULawT_fn(clean_wav_batch, 255.0)
    clean_normed_stft_batch = clean_wav_features.normed_stft_batch

    est_mag_batch_FT = self.variables.FeatureTransformer(est_mag_batch)
    clean_mag_batch_FT = self.variables.FeatureTransformer(clean_mag_batch)


    # region losses
    self.loss_compressedMag_mse = losses.FSum_compressedMag_mse(
        est_mag_batch, clean_mag_batch, PARAM.loss_compressedMag_idx)
    self.loss_compressedStft_mse = losses.FSum_compressedStft_mse(
        est_mag_batch, est_normed_stft_batch,
        clean_mag_batch, clean_normed_stft_batch,
        PARAM.loss_compressedMag_idx)


    self.loss_mag_mse = losses.FSum_MSE(est_mag_batch, clean_mag_batch)
    self.loss_mag_reMse = losses.FSum_relativeMSE(est_mag_batch, clean_mag_batch,
                                                  PARAM.relative_loss_epsilon, PARAM.RL_idx)
    self.loss_stft_mse = losses.FSum_MSE(est_stft_batch, clean_stft_batch)
    self.loss_stft_reMse = losses.FSum_relativeMSE(est_stft_batch, clean_stft_batch,
                                                   PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_mag_mae = losses.FSum_MAE(est_mag_batch, clean_mag_batch)
    self.loss_mag_reMae = losses.FSum_relativeMAE(est_mag_batch, clean_mag_batch,
                                                  PARAM.relative_loss_epsilon)
    self.loss_stft_mae = losses.FSum_MAE(est_stft_batch, clean_stft_batch)
    self.loss_stft_reMae = losses.FSum_relativeMAE(est_stft_batch, clean_stft_batch,
                                                   PARAM.relative_loss_epsilon)


    self.loss_wav_L1 = losses.FSum_MAE(est_wav_batch, clean_wav_batch)
    self.loss_wav_L2 = losses.FSum_MSE(est_wav_batch, clean_wav_batch)
    self.loss_wav_reL2 = losses.FSum_relativeMSE(est_wav_batch, clean_wav_batch,
                                                 PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_CosSim = losses.batchMean_CosSim_loss(est_wav_batch, clean_wav_batch)
    self.loss_SquareCosSim = losses.batchMean_SquareCosSim_loss(
          est_wav_batch, clean_wav_batch)
    self.loss_stCosSim = losses.batchMean_short_time_CosSim_loss(
        est_wav_batch, clean_wav_batch,
        PARAM.st_frame_length_for_loss,
        PARAM.st_frame_step_for_loss)
    self.loss_stSquareCosSim = losses.batchMean_short_time_SquareCosSim_loss(
        est_wav_batch, clean_wav_batch,
        PARAM.st_frame_length_for_loss,
        PARAM.st_frame_step_for_loss)

    self.FTloss_mag_mse = losses.FSum_MSE(est_mag_batch_FT, clean_mag_batch_FT)
    self.FTloss_mag_mae = losses.FSum_MAE(est_mag_batch_FT, clean_mag_batch_FT)
    self.loss_ulawCosSim = losses.batchMean_CosSim_loss(est_wav_ulawWav_batch, clean_wav_ulawWav_batch)

    self.d_loss = tf.losses.softmax_cross_entropy(d_labels, d_logits)

    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_mag_mae': self.loss_mag_mae,
        'loss_mag_reMae': self.loss_mag_reMae,
        'loss_stft_mae': self.loss_stft_mae,
        'loss_stft_reMae': self.loss_stft_reMae,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
        'loss_stCosSim': self.loss_stCosSim,
        'loss_stSquareCosSim': self.loss_stSquareCosSim,
        'FTloss_mag_mse': self.FTloss_mag_mse,
        'FTloss_mag_mae': self.FTloss_mag_mae,
        'loss_ulawCosSim': self.loss_ulawCosSim,
        'd_loss': self.d_loss,
    }
    # endregion losses

    # region sum_loss_G
    sum_loss_G = 0.0
    sum_loss_names = PARAM.sum_losses_G
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_G_w) > 0:
        loss_t *= PARAM.sum_losses_G_w[i]
      sum_loss_G += loss_t
    # endregion sum_loss_G

    # region sum_loss_D
    sum_loss_D = 0.0
    sum_loss_names = PARAM.sum_losses_D
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_D_w) > 0:
        loss_t *= PARAM.sum_losses_D_w[i]
      sum_loss_D += loss_t
    # endregion sum_loss_D

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t *= PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = tf.stack(show_losses)
    # endregion show_losses

    # region stop_criterion_losses
    stop_criterion_losses_sum = 0.0
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t *= PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t
    # endregion stop_criterion_losses

    return Losses(sum_loss_G=sum_loss_G,
                  sum_loss_D=sum_loss_D,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)


  def change_lr(self, sess, new_lr):
    sess.run(self.assign_lr, feed_dict={self.new_lr:new_lr})

  @property
  def global_step(self):
    return self._global_step

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_features.wav_batch

  @property
  def train_op(self):
    return self._train_op

  @property
  def losses(self):
    return self._losses

  @property
  def lr(self):
    return self._lr

  @property
  def est_clean_wav_batch(self):
    return self.est_wav_features.wav_batch
