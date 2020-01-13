import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from .modules import RealVariables

class DISCRIMINATOR_AD_MODEL(Module):
  def __init__(self,
               mode,
               variables: RealVariables,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    super(DISCRIMINATOR_AD_MODEL, self).__init__(
        mode,
        variables,
        mixed_wav_batch,
        clean_wav_batch,
        noise_wav_batch)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    ## se not_transformed_losses grads
    se_NTloss_grads = tf.gradients(
      self._not_transformed_losses,
      self.se_net_params,
      colocate_gradients_with_ops=True
    )
    se_NTloss_grads, _ = tf.clip_by_global_norm(se_NTloss_grads, PARAM.max_gradient_norm)

    ## se transformed_losses grads
    se_Tloss_grads = tf.gradients(
      self._transformed_losses,
      self.se_net_params,
      colocate_gradients_with_ops=True
    )
    se_Tloss_grads, _ = tf.clip_by_global_norm(se_Tloss_grads, PARAM.max_gradient_norm)


    ## discriminator loss grads in D_net 判别器损失对判别网络的梯度
    d_grads_in_D_Net = tf.gradients(
      self._d_loss,
      self.d_params,
      colocate_gradients_with_ops=True
    )
    d_grads_in_D_Net, _ = tf.clip_by_global_norm(d_grads_in_D_Net, PARAM.max_gradient_norm)
    d_grads_in_D_Net = [grad*PARAM.discirminator_grad_coef for grad in d_grads_in_D_Net]

    ## feature transformer grads
    d_grads_in_FT = tf.gradients(
      self._d_loss,
      self.ft_params,
      colocate_gradients_with_ops=True
    )
    d_grads_in_FT, _ = tf.clip_by_global_norm(d_grads_in_FT, PARAM.max_gradient_norm)
    d_grads_in_FT = [grad*PARAM.feature_transformer_grad_coef for grad in d_grads_in_FT]

    all_grads = []
    all_params = []

    if "not_transformed_losses" in PARAM.losses_position:
      all_grads = se_NTloss_grads
      all_params = self.se_net_params

    if "transformed_losses" in PARAM.losses_position:
      ## FT_grad in se_net
      if len(all_grads)==0:
        all_grads = se_Tloss_grads
        all_params = self.se_net_params
      else:
        # merge se_grads from se_loss and FT_grad fron FT_loss
        all_grads = [(grad1+grad2)*0.5 for grad1, grad2 in zip(all_grads, se_Tloss_grads)]

      ## d_grad in D_net
      if PARAM.discirminator_grad_coef > 1e-12:
        print('optimizer D')
        # merge d_grads_in_D_Net and D_params
        all_grads = all_grads + d_grads_in_D_Net
        all_params = all_params + self.d_params

      ## d_grads in Feature Transformer
      if PARAM.feature_transformer_grad_coef > 1e-12:
        print('optimizer feature transformer')
        # merge d_grads_in_FT and FT_params
        all_grads = all_grads + d_grads_in_FT
        all_params = all_params + self.ft_params


    assert len(all_grads)>0, "Losses are all turned off."


    all_clipped_grads, _ = tf.clip_by_global_norm(all_grads, PARAM.max_gradient_norm)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(self._lr)
    self._train_op = self.optimizer.apply_gradients(zip(all_clipped_grads, all_params),
                                                    global_step=self.global_step)


  def forward(self, mixed_wav_batch_extend):
    r_outputs = self.real_networks_forward(mixed_wav_batch_extend)
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = r_outputs

    return r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch


  def get_discriminator_loss(self, forward_outputs):
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = forward_outputs
    logits, one_hots_labels, deep_features = self.clean_and_enhanced_mag_discriminator(self.clean_mag_batch, r_est_clean_mag_batch)
    # print("23333333333333", one_hots_labels.shape.as_list(), logits.shape.as_list())
    loss = tf.losses.softmax_cross_entropy(one_hots_labels, logits) # max about 0.7
    loss = loss*PARAM.D_loss_coef
    return loss


  def get_not_transformed_loss(self, forward_outputs):
    clean_mag_batch_label = self.clean_mag_batch
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = forward_outputs

    if PARAM.use_noLabel_noisy_speech: # delete noLabel data
      r_est_clean_mag_batch, _ = tf.split(r_est_clean_mag_batch,
                                          [self.label_noisy_batchsize, self.noLabel_noisy_batchsize], axis=0)
      r_est_clean_spec_batch, _ = tf.split(r_est_clean_spec_batch,
                                           [self.label_noisy_batchsize, self.noLabel_noisy_batchsize], axis=0)
      r_est_clean_wav_batch, _ = tf.split(r_est_clean_wav_batch,
                                          [self.label_noisy_batchsize, self.noLabel_noisy_batchsize], axis=0)

    # not_transformed_losses
    self.loss_mag_mse = losses.batch_time_fea_real_mse(r_est_clean_mag_batch, clean_mag_batch_label)
    self.loss_reMagMse = losses.batch_real_relativeMSE(r_est_clean_mag_batch, clean_mag_batch_label,
                                                       PARAM.relative_loss_epsilon)
    self.loss_spec_mse = losses.batch_time_fea_complex_mse(r_est_clean_spec_batch, self.clean_spec_batch)
    self.loss_reSpecMse = losses.batch_complex_relativeMSE(r_est_clean_spec_batch, self.clean_spec_batch,
                                                           PARAM.relative_loss_epsilon)

    self.loss_wav_L1 = losses.batch_wav_L1_loss(r_est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.loss_wav_L2 = losses.batch_wav_L2_loss(r_est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.loss_reWavL2 = losses.batch_wav_relativeMSE(r_est_clean_wav_batch, self.clean_wav_batch, PARAM.relative_loss_epsilon)
    self.loss_sdrV1 = losses.batch_sdrV1_loss(r_est_clean_wav_batch, self.clean_wav_batch)
    self.loss_sdrV2 = losses.batch_sdrV2_loss(r_est_clean_wav_batch, self.clean_wav_batch)
    self.loss_cosSimV1 = losses.batch_cosSimV1_loss(r_est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.loss_cosSimV2 = losses.batch_cosSimV2_loss(r_est_clean_wav_batch, self.clean_wav_batch) # *0.334
    self.loss_stSDRV3 = losses.batch_short_time_sdrV3_loss(r_est_clean_wav_batch, self.clean_wav_batch,
                                                           PARAM.st_frame_length_for_loss,
                                                           PARAM.st_frame_step_for_loss)
    # engregion losses

    loss = 0
    loss_names = PARAM.not_transformed_losses

    for i, name in enumerate(loss_names):
      loss_t = {
        'loss_mag_mse': self.loss_mag_mse,
        'loss_reMagMse': self.loss_reMagMse,
        'loss_spec_mse': self.loss_spec_mse,
        'loss_reSpecMse': self.loss_reSpecMse,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_reWavL2': self.loss_reWavL2,
        'loss_sdrV1': self.loss_sdrV1,
        'loss_sdrV2': self.loss_sdrV2,
        'loss_cosSimV1': self.loss_cosSimV1,
        'loss_cosSimV2': self.loss_cosSimV2,
        'loss_stSDRV3': self.loss_stSDRV3,
      }[name]
      if len(PARAM.NTloss_weight) > 0:
        loss_t *= PARAM.NTloss_weight[i]
      loss += loss_t
    return loss


  def get_transformed_loss(self, forward_outputs):
    clean_mag_batch_label = self.clean_mag_batch
    r_est_clean_mag_batch, _, _ = forward_outputs
    if PARAM.use_noLabel_noisy_speech: # delete noLabel data
      r_est_clean_mag_batch, _ = tf.split(r_est_clean_mag_batch,
                                          [self.label_noisy_batchsize, self.noLabel_noisy_batchsize], axis=0)

    # feature transformer
    clean_mag_batch_label = self.variables.FeatureTransformer(clean_mag_batch_label)
    r_est_clean_mag_batch = self.variables.FeatureTransformer(r_est_clean_mag_batch)

    # not_transformed_losses
    self.FTloss_mag_mse = losses.batch_time_fea_real_mse(r_est_clean_mag_batch, clean_mag_batch_label)
    self.FTloss_mag_RL = losses.batch_real_relativeMSE(r_est_clean_mag_batch, clean_mag_batch_label,
                                                       PARAM.relative_loss_epsilon)
    # engregion losses

    loss = 0
    loss_names = PARAM.transformed_losses

    for i, name in enumerate(loss_names):
      loss_t = {
        'FTloss_mag_mse': self.FTloss_mag_mse,
        'FTloss_mag_RL': self.FTloss_mag_RL,
      }[name]
      if len(PARAM.Tloss_weight) > 0:
        loss_t *= PARAM.Tloss_weight[i]
      loss += loss_t
    return loss

  def get_loss(self, forward_outputs):
    _not_transformed_losses = self.get_not_transformed_loss(forward_outputs)
    _transformed_losses = self.get_transformed_loss(forward_outputs)
    _d_loss = self.get_discriminator_loss(forward_outputs)

    if PARAM.weighted_FTL_by_DLoss:
      # w_FTL_ref_DLoss = 1.0/(1.0+tf.exp(tf.stop_gradient(loss)*40.0-4.0))
      w_FTL_ref_DLoss = tf.nn.sigmoid(4.0-tf.stop_gradient(_d_loss)*PARAM.D_strict_degree_for_FTL)
      _transformed_losses = _transformed_losses * w_FTL_ref_DLoss

    return _not_transformed_losses, _transformed_losses, _d_loss
