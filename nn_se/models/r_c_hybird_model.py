import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses
from .modules import RCHybirdVariables

class RC_HYBIRD_MODEL(Module):
  def __init__(self,
               mode,
               variables: RCHybirdVariables,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    super(RC_HYBIRD_MODEL, self).__init__(
        mode,
        variables,
        mixed_wav_batch,
        clean_wav_batch,
        noise_wav_batch)

  def forward(self, mixed_wav_batch):
    r_outputs = self.real_networks_forward(mixed_wav_batch)
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = r_outputs

    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]

    c_outputs = self.post_complex_networks_forward(r_est_clean_mag_batch, self.mixed_angle_batch, _mixed_wav_len)
    c_est_clean_mag_batch, c_est_clean_spec_batch, c_est_clean_wav_batch = c_outputs
    return (r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch,
            c_est_clean_mag_batch, c_est_clean_spec_batch, c_est_clean_wav_batch)

  def get_loss(self, forward_outputs):
    (r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch,
     c_est_clean_mag_batch, c_est_clean_spec_batch, c_est_clean_wav_batch) = forward_outputs

    # region real net losses
    ## frequency domain loss
    self.real_net_mag_mse = losses.batch_time_fea_real_mse(r_est_clean_mag_batch, self.clean_mag_batch)
    self.real_net_reMagMse = losses.batch_real_relativeMSE(r_est_clean_mag_batch, self.clean_mag_batch, PARAM.relative_loss_epsilon)
    self.real_net_spec_mse = losses.batch_time_fea_complex_mse(r_est_clean_spec_batch, self.clean_spec_batch)
    self.real_net_reSpecMse = losses.batch_complex_relativeMSE(r_est_clean_spec_batch, self.clean_spec_batch, PARAM.relative_loss_epsilon)
    self.real_net_specTCosSimV1 = losses.batch_complexspec_timeaxis_cos_sim_V1(r_est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specFCosSimV1 = losses.batch_complexspec_frequencyaxis_cos_sim_V1(r_est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specTFCosSimV1 = losses.batch_complexspec_TF_cos_sim_V1(r_est_clean_spec_batch, self.clean_spec_batch) # *0.167

    ## time domain loss
    self.real_net_wav_L1 = losses.batch_wav_L1_loss(r_est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.real_net_wav_L2 = losses.batch_wav_L2_loss(r_est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.real_net_reWavL2 = losses.batch_wav_relativeMSE(r_est_clean_wav_batch, self.clean_wav_batch, PARAM.relative_loss_epsilon)
    self.real_net_sdrV1 = losses.batch_sdrV1_loss(r_est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV2 = losses.batch_sdrV2_loss(r_est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV3 = losses.batch_sdrV3_loss(r_est_clean_wav_batch, self.clean_wav_batch, PARAM.sdrv3_bias) # *0.167
    if PARAM.sdrv3_bias:
      assert PARAM.sdrv3_bias > 0.0, 'sdrv3_bias is constrained larger than zero. _real'
      self.real_net_sdrV3 *= 1.0 + 60 * PARAM.sdrv3_bias
    self.real_net_cosSimV1 = losses.batch_cosSimV1_loss(r_est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.real_net_cosSimV1WT10 = self.real_net_cosSimV1*0.167 # loss weighted to 10 level
    self.real_net_cosSimV2 = losses.batch_cosSimV2_loss(r_est_clean_wav_batch, self.clean_wav_batch) # *0.334
    self.real_net_stSDRV3 = losses.batch_short_time_sdrV3_loss(r_est_clean_wav_batch, self.clean_wav_batch,
                                                               PARAM.st_frame_length_for_loss,
                                                               PARAM.st_frame_step_for_loss)
    # engregion losses

    # region real net losses
    ## frequency domain loss
    self.comp_net_mag_mse = losses.batch_time_fea_real_mse(c_est_clean_mag_batch, self.clean_mag_batch)
    self.comp_net_reMagMse = losses.batch_real_relativeMSE(c_est_clean_mag_batch, self.clean_mag_batch, PARAM.relative_loss_epsilon)

    # TODO: testing
    cliped_clean_spec_batch = self.clean_spec_batch
    if PARAM.complex_clip_label_mag:
      cliped_clean_mag_batch = tf.clip_by_value(self.clean_mag_batch, 0.0, float(PARAM.complex_clip_mag_max))
      cliped_clean_spec_batch = tf.complex(cliped_clean_mag_batch, 0.0) * tf.exp(tf.complex(0.0, tf.angle(self.clean_spec_batch)))

    self.comp_net_spec_mse = losses.batch_time_fea_complex_mse(c_est_clean_spec_batch, cliped_clean_spec_batch)
    self.comp_net_reSpecMse = losses.batch_complex_relativeMSE(c_est_clean_spec_batch, cliped_clean_spec_batch, PARAM.relative_loss_epsilon)
    self.comp_net_specTCosSimV1 = losses.batch_complexspec_timeaxis_cos_sim_V1(c_est_clean_spec_batch, cliped_clean_spec_batch) # *0.167
    self.comp_net_specFCosSimV1 = losses.batch_complexspec_frequencyaxis_cos_sim_V1(c_est_clean_spec_batch, cliped_clean_spec_batch) # *0.167
    self.comp_net_specTFCosSimV1 = losses.batch_complexspec_TF_cos_sim_V1(c_est_clean_spec_batch, cliped_clean_spec_batch) # *0.167

    ## time domain loss
    self.comp_net_wav_L1 = losses.batch_wav_L1_loss(c_est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.comp_net_wav_L2 = losses.batch_wav_L2_loss(c_est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.comp_net_reWavL2 = losses.batch_wav_relativeMSE(c_est_clean_wav_batch, self.clean_wav_batch, PARAM.relative_loss_epsilon)
    self.comp_net_sdrV1 = losses.batch_sdrV1_loss(c_est_clean_wav_batch, self.clean_wav_batch)
    self.comp_net_sdrV2 = losses.batch_sdrV2_loss(c_est_clean_wav_batch, self.clean_wav_batch)
    self.comp_net_sdrV3 = losses.batch_sdrV3_loss(c_est_clean_wav_batch, self.clean_wav_batch, PARAM.sdrv3_bias) # *0.167
    if PARAM.sdrv3_bias:
      assert PARAM.sdrv3_bias > 0.0, 'sdrv3_bias is constrained larger than zero. _real'
      self.comp_net_sdrV3 *= 1.0 + 60 * PARAM.sdrv3_bias
    self.comp_net_cosSimV1 = losses.batch_cosSimV1_loss(c_est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.comp_net_cosSimV1WT10 = self.comp_net_cosSimV1*0.167 # loss weighted to 10 level
    self.comp_net_cosSimV2 = losses.batch_cosSimV2_loss(c_est_clean_wav_batch, self.clean_wav_batch) # *0.334
    self.comp_net_stSDRV3 = losses.batch_short_time_sdrV3_loss(c_est_clean_wav_batch, self.clean_wav_batch,
                                                               PARAM.st_frame_length_for_loss,
                                                               PARAM.st_frame_step_for_loss)
    # engregion losses

    loss = 0
    loss_names = PARAM.loss_name

    for i, name in enumerate(loss_names):
      loss_t = {
        'real_net_mag_mse': self.real_net_mag_mse,
        'real_net_reMagMse': self.real_net_reMagMse,
        'real_net_spec_mse': self.real_net_spec_mse,
        'real_net_reSpecMse': self.real_net_reSpecMse,
        'real_net_wav_L1': self.real_net_wav_L1,
        'real_net_wav_L2': self.real_net_wav_L2,
        'real_net_reWavL2': self.real_net_reWavL2,
        'real_net_sdrV1': self.real_net_sdrV1,
        'real_net_sdrV2': self.real_net_sdrV2,
        'real_net_sdrV3': self.real_net_sdrV3,
        'real_net_cosSimV1': self.real_net_cosSimV1,
        'real_net_cosSimV1WT10': self.real_net_cosSimV1WT10,
        'real_net_cosSimV2': self.real_net_cosSimV2,
        'real_net_specTCosSimV1': self.real_net_specTCosSimV1,
        'real_net_specFCosSimV1': self.real_net_specFCosSimV1,
        'real_net_specTFCosSimV1': self.real_net_specTFCosSimV1,
        'real_net_stSDRV3': self.real_net_stSDRV3,

        'comp_net_mag_mse': self.comp_net_mag_mse,
        'comp_net_reMagMse': self.comp_net_reMagMse,
        'comp_net_spec_mse': self.comp_net_spec_mse,
        'comp_net_reSpecMse': self.comp_net_reSpecMse,
        'comp_net_wav_L1': self.comp_net_wav_L1,
        'comp_net_wav_L2': self.comp_net_wav_L2,
        'comp_net_reWavL2': self.comp_net_reWavL2,
        'comp_net_sdrV1': self.comp_net_sdrV1,
        'comp_net_sdrV2': self.comp_net_sdrV2,
        'comp_net_sdrV3': self.comp_net_sdrV3,
        'comp_net_cosSimV1': self.comp_net_cosSimV1,
        'comp_net_cosSimV1WT10': self.comp_net_cosSimV1WT10,
        'comp_net_cosSimV2': self.comp_net_cosSimV2,
        'comp_net_specTCosSimV1': self.comp_net_specTCosSimV1,
        'comp_net_specFCosSimV1': self.comp_net_specFCosSimV1,
        'comp_net_specTFCosSimV1': self.comp_net_specTFCosSimV1,
        'comp_net_stSDRV3': self.comp_net_stSDRV3,
      }[name]
      if len(PARAM.loss_weight) > 0:
        loss_t *= PARAM.loss_weight[i]
      loss += loss_t
    return loss
