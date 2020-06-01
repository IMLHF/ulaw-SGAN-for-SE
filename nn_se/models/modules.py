import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self._global_step = self.add_variable('global_step', dtype=tf.int32,
                                          initializer=tf.constant_initializer(1), trainable=False)
    self._lr = self.add_variable('lr', dtype=tf.float32, trainable=False,
                                 initializer=tf.constant_initializer(PARAM.learning_rate))

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
      dropL = tf.keras.layers.Dropout(rate=PARAM.blstm_drop_rate)
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

    self._f_u = tf.constant(-1.0)
    if PARAM.add_ulawFT_in_G:
      # belong to discriminator
      self._f_u_var = self.add_variable('G/FTL/f_u', shape=[PARAM.n_u_var], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01),
                                        trainable=PARAM.f_u_var_trainable)
      self._f_u = PARAM.u_eps + tf.abs(tf.reduce_sum(self._f_u_var))*PARAM.u_times

      def ulaw_fn(x):
        # x: [batch, time, fea]
        u = self._f_u
        y = tf.log(x * u + 1.0) / tf.log(u + 1.0)
        return y
      self.ulaw_fn = ulaw_fn

  def call(self, input_feature_batch, training=False):
    outputs = input_feature_batch # [batch, time, feature_dim]
    _batch_size = tf.shape(outputs)[0]

    if PARAM.add_ulawFT_in_G:
      outputs = self.ulaw_fn(outputs)

    # BLSTM
    # self.blstm_outputs = []
    for blstm in self.blstm_layers:
      outputs = blstm(outputs, training=training)
      # self.blstm_outputs.append(outputs)

    # LSTM
    for lstm in self.lstm_layers:
      outputs = lstm(outputs, training=training)

    # FC
    if len(self.blstm_layers) > 0 and len(self.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.N_RNN_CELL])
    outputs = self.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.feature_dim])
    return outputs

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    #### discriminator
    self.N_RNN_CELL = PARAM.rnn_units
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
      dropL = tf.keras.layers.Dropout(rate=PARAM.blstm_drop_rate)
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

    self.D_out_fc = tf.keras.layers.Dense(1, name='D/out_fc')

    ### FeatureTransformerLayers
    if "trainableUlaw_v2" in PARAM.FT_type:
      # belong to discriminator
      self._f_u_var = self.add_variable('D/FTL/f_u', shape=[PARAM.n_u_var], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01),
                                        trainable=PARAM.f_u_var_trainable)
      self._f_u = PARAM.u_eps + tf.abs(tf.reduce_sum(self._f_u_var))*PARAM.u_times

    def ulaw_fn(x):
      # x: [batch, time, fea]
      u = self._f_u
      y = tf.log(x * u + 1.0) / tf.log(u + 1.0)
      return y
    self.ulaw_fn = ulaw_fn

    if "dense" in PARAM.FT_type:
      ## RandomDenseT
      self.RandomDenseT = tf.keras.layers.Dense(self.N_RNN_CELL, activation='tanh',
                                                name='D/FTL/FT_Dense')
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
        if ft_type == "trainableUlaw_v2":
          x = self.ulaw_fn(x)
        elif ft_type == "dense":
          x = self.RandomDenseT(x)
        # elif ft_type == "MelDenseT":
        #   x = self.MelDenseT(x)
        else:
          raise NotImplementedError
      return x
    self.FeatureTransformer = FeatureTransformer

  def call(self, est_mag_batch, clean_mag_batch, training):
    est_fea_batch = est_mag_batch
    clean_fea_batch = clean_mag_batch
    outputs = tf.concat([est_fea_batch, clean_fea_batch], axis=0)

    # Features Transformer
    outputs = self.FeatureTransformer(outputs)
    _batch_size = outputs.shape[0]

    # if PARAM.frame_level_D:
    #   zeros = tf.zeros(est_fea_batch.shape[0:2], dtype=tf.int32)
    #   ones = tf.ones(clean_fea_batch.shape[0:2], dtype=tf.int32)
    # else:
    #   zeros = tf.zeros(est_fea_batch.shape[0], dtype=tf.int32)
    #   ones = tf.ones(clean_fea_batch.shape[0], dtype=tf.int32)
    # labels = tf.concat([zeros, ones], axis=0)

    for blstm in self.D_blstm_layers:
      outputs = blstm(outputs, training=training)

    for lstm in self.D_lstm_layers:
      outputs = lstm(outputs, training=training)

    if not PARAM.frame_level_D:
      outputs = outputs[:,-1,:]

    # FC
    if len(self.D_blstm_layers) > 0 and len(self.D_lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.N_RNN_CELL])
    outputs = self.D_out_fc(outputs) # [N, T, 1] or [N, 1]
    if PARAM.frame_level_D:
      outputs = tf.reshape(outputs, [_batch_size, -1, 1])
    # outputs = tf.keras.activations.sigmoid(outputs) # mv to losses
    clean_d_out, est_d_out = tf.split(outputs, 2, axis=0)
    return clean_d_out, est_d_out # [N, 1], [N, 1] or [N,T,1],[N,T,1]


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
               generator: Union[Generator],
               discriminator: Union[Discriminator],
               mixed_wav_batch,
               clean_wav_batch=None):
    self.mode = mode
    self.generator = generator
    self.discriminator = discriminator
    mixed_stft_batch = misc_utils.tf_wav2stft(mixed_wav_batch, PARAM.frame_length,
                                              PARAM.frame_step, PARAM.fft_length)  # [N, T, F]
    mixed_mag_batch = tf.abs(mixed_stft_batch) # [N, F, T]
    mixed_angle_batch = tf.angle(mixed_stft_batch)
    if PARAM.stft_norm_method == 'polar':
      mixed_normed_stft_batch = tf.exp(tf.complex(0.0, mixed_angle_batch))
    elif PARAM.stft_norm_method == 'div':
      mixed_normed_stft_batch = tf.divide(mixed_stft_batch, tf.complex(mixed_mag_batch+1e-7, 0.0))
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
      if PARAM.stft_norm_method == 'polar':
        clean_normed_stft_batch = tf.exp(tf.complex(0.0, clean_angle_batch))
      elif PARAM.stft_norm_method == 'div':
        clean_normed_stft_batch = tf.divide(clean_stft_batch, tf.complex(clean_mag_batch+1e-7, 0.0))
      self.clean_wav_features = WavFeatures(wav_batch=clean_wav_batch,
                                            stft_batch=clean_stft_batch,
                                            mag_batch=clean_mag_batch,
                                            angle_batch=clean_angle_batch,
                                            normed_stft_batch=clean_normed_stft_batch)

    # nn_se forward
    self.est_wav_features = self._forward()

    if self.mode != PARAM.MODEL_INFER_KEY:
      self._clean_d_out, self._est_d_out = None, None
      training = (self.mode == PARAM.MODEL_TRAIN_KEY)
      self._clean_d_out, self._est_d_out = self.discriminator(self.est_wav_features.mag_batch,
                                                              self.clean_wav_features.mag_batch,
                                                              training)

      # losses
      self._losses = self._get_losses(self.est_wav_features, self.clean_wav_features,
                                      self._clean_d_out, self._est_d_out)

    # global_step, lr, vars
    self._global_step = self.generator._global_step
    self._lr = self.generator._lr

    # for lr halving
    self.new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self.assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self._global_step, warmup_steps=PARAM.warmup_steps)

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.d_params = self.discriminator.trainable_variables
    self.g_params = self.generator.trainable_variables
    self.save_variables = self.generator.variables + self.discriminator.variables
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=40,
                                          save_relative_paths=True)

    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nD Trainable PARAMs:")
      misc_utils.show_variables(self.d_params)
      print("\nG Trainable PARAMs")
      misc_utils.show_variables(self.g_params)
      # print("\n save_vars")
      # misc_utils.show_variables(self.save_variables)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    if PARAM.optimizer == "Adam":
      self.g_optimizer = tf.keras.optimizers.Adam(self._lr)
      self.d_optimizer = tf.keras.optimizers.Adam(self._lr)
    elif PARAM.optimizer == "RMSProp":
      self.g_optimizer = tf.keras.optimizers.RMSProp(self._lr)
      self.d_optimizer = tf.keras.optimizers.RMSProp(self._lr)

    ## G grads
    grads_G = self.g_optimizer.get_gradients(
      self._losses.sum_loss_G,
      self.g_params,
    )
    grads_G, _ = tf.clip_by_global_norm(grads_G, PARAM.max_gradient_norm)
    self._train_op_G = self.g_optimizer.apply_gradients(zip(grads_G, self.g_params))

    ## D grads
    if len(PARAM.sum_losses_D)>0:
      grads_D = self.d_optimizer.get_gradients(
        self._losses.sum_loss_D,
        self.d_params,
      )
      grads_D, _ = tf.clip_by_global_norm(grads_D, PARAM.max_gradient_norm)
      self._train_op_D = self.d_optimizer.apply_gradients(zip(grads_D, self.d_params))
    else:
      self._train_op_D = tf.no_op()

    self._global_step_increase = self._global_step.assign_add(1)
    self._train_op = tf.group(self._train_op_D, self._train_op_G)
    # self.adam_p = self.optimizer.variables()
    # for p in self.adam_p:
    #   print(p)


  def _forward(self):
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    nn_out = self.generator(self.mixed_wav_features.mag_batch, training)

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


  def _get_losses(self, est_wav_features, clean_wav_features,
                  clean_d_out, est_d_out):

    def FixULawT_fn(x, u):
      # x: [batch, time, fea]
      y = tf.sign(x) * tf.log(tf.abs(x) * u + 1.0) / tf.log(u + 1.0)
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

    est_mag_batch_FT = self.discriminator.FeatureTransformer(est_mag_batch)
    clean_mag_batch_FT = self.discriminator.FeatureTransformer(clean_mag_batch)


    # region losses
    self.loss_compressedMag_mse = losses.FSum_compressedMag_mse(
        est_mag_batch, clean_mag_batch, PARAM.loss_compressedMag_idx)
    self.loss_compressedStft_mse = losses.FSum_compressedStft_mse(
        est_mag_batch, est_normed_stft_batch,
        clean_mag_batch, clean_normed_stft_batch,
        PARAM.loss_compressedMag_idx)


    self.loss_logmag_mse = losses.FSum_MSE(tf.log(est_mag_batch+1e-12), tf.log(clean_mag_batch+1e-12))
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


    self.loss_ssnr = losses.batchMean_SSNR(est_wav_batch, clean_wav_batch)
    self.loss_cssnr = losses.batchMean_CSSNR(est_wav_batch, clean_wav_batch)
    self.loss_wav_L1 = losses.FSum_MAE(est_wav_batch, clean_wav_batch)
    self.loss_wav_L2 = losses.FSum_MSE(est_wav_batch, clean_wav_batch)
    self.loss_ulawwav_L1 = losses.FSum_MAE(tf.expand_dims(est_wav_ulawWav_batch, -1),
                                           tf.expand_dims(clean_wav_ulawWav_batch, -1))
    self.loss_wav_reL2 = losses.FSum_relativeMSE(est_wav_batch, clean_wav_batch,
                                                 PARAM.relative_loss_epsilon, PARAM.RL_idx)
    self.loss_ulawCosSim = losses.batchMean_CosSim_loss(est_wav_ulawWav_batch, clean_wav_ulawWav_batch)

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

    self.d_loss = losses.d_loss(clean_d_out, est_d_out)
    self.d_loss_rasgan = losses.d_loss_rasgan(clean_d_out, est_d_out)

    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_logmag_mse': self.loss_logmag_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_mag_mae': self.loss_mag_mae,
        'loss_mag_reMae': self.loss_mag_reMae,
        'loss_stft_mae': self.loss_stft_mae,
        'loss_stft_reMae': self.loss_stft_reMae,
        'loss_ssnr': self.loss_ssnr,
        'loss_cssnr': self.loss_ssnr,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_ulawwav_L1': self.loss_ulawwav_L1,
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
        'd_loss_rasgan': self.d_loss_rasgan,
    }
    # endregion losses

    # region sum_loss_G
    sum_loss_G = 0.0
    sum_loss_names = PARAM.sum_losses_G
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_G_w) > 0:
        loss_t = loss_t*PARAM.sum_losses_G_w[i]
      sum_loss_G += loss_t
    # sum_loss_G = sum_loss_G - self.d_loss
    # endregion sum_loss_G

    # region sum_loss_D
    sum_loss_D = tf.constant(0.0)
    sum_loss_names = PARAM.sum_losses_D
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_D_w) > 0:
        loss_t = loss_t*PARAM.sum_losses_D_w[i]
      sum_loss_D += loss_t
    # endregion sum_loss_D

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t = loss_t * PARAM.show_losses_w[i]
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
