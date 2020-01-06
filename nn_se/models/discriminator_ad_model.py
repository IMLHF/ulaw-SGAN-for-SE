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

    ## se_loss grads 增强网络的损失的梯度，例如MSE
    se_loss_grads = self.se_loss_grads

    ## deep features loss grads 判别器中deepFeatureLoss对增强网络求导的梯度
    deep_f_loss_grads = tf.gradients(
      self._deep_features_loss,
      self.se_net_params,
      colocate_gradients_with_ops=True
    )
    deep_f_loss_grads, _ = tf.clip_by_global_norm(deep_f_loss_grads, PARAM.max_gradient_norm)

    ## discriminator loss grads in se_net 判别器损失对增强网络的梯度
    d_grads_in_seNet = tf.gradients(
      self._d_loss,
      self.se_net_params,
      colocate_gradients_with_ops=True
    )
    d_grads_in_seNet, _ = tf.clip_by_global_norm(d_grads_in_seNet, PARAM.max_gradient_norm)

    ## discriminator loss grads in D_net 判别器损失对判别网络的梯度
    d_grads_in_D_Net = tf.gradients(
      self._d_loss,
      self.d_params,
      colocate_gradients_with_ops=True
    )
    d_grads_in_D_Net, _ = tf.clip_by_global_norm(d_grads_in_D_Net, PARAM.max_gradient_norm)

    # region d_grads_in_seNet
    # ifD_passGrad_to_SE = tf.cast(tf.bitwise.bitwise_and(self.global_step//2250, 1), tf.float32) # Alternate Training for GANs
    ifD_passGrad_to_SE = 1.0
    d_grads_in_seNet = [grad*PARAM.se_grad_fromD_coef*ifD_passGrad_to_SE for grad in d_grads_in_seNet]
    if PARAM.D_GRL and PARAM.se_grad_fromD_coef*ifD_passGrad_to_SE > 1e-12:
      d_grads_in_seNet = [-grad for grad in d_grads_in_seNet] # GRL
    if PARAM.D_Grad_DCC and PARAM.se_grad_fromD_coef*ifD_passGrad_to_SE > 1e-12: # Direction Consistent Constraints
      ## D_GRL_005
      # d_grads_in_seNet = [
      #     tf.expand_dims(tf.nn.relu(tf.reduce_sum(grad1*grad2,-1)/tf.reduce_sum(grad1*grad1, -1)), -1)*grad1 for grad1, grad2 in zip(se_loss_grads, d_grads_in_seNet)]

      ## D_GRL_006
      # constrainted_se_grads_fromD = []
      # for grad1, grad2 in zip(se_loss_grads, d_grads_in_seNet):
      #   w_of_grad2 = (1+tf.abs(tf.sign(grad1)+tf.sign(grad2))) // 2
      #   constrainted_grad2 = w_of_grad2 * grad2
      #   constrainted_se_grads_fromD.append(constrainted_grad2)
      # d_grads_in_seNet = constrainted_se_grads_fromD

      ## D_GRL_007
      constrainted_se_grads_fromD = []
      for grad1, grad2 in zip(se_loss_grads, d_grads_in_seNet):
        grad_shape = grad1.shape.as_list()
        vec1 = tf.reshape(grad1,[-1])
        vec2 = tf.reshape(grad2,[-1])
        prj_on_vec1 = tf.nn.relu(tf.reduce_sum(vec1*vec2,-1)/tf.reduce_sum(vec1*vec1, -1))*vec1
        constrainted_grad2 = tf.reshape(prj_on_vec1, grad_shape)
        constrainted_se_grads_fromD.append(constrainted_grad2)
      d_grads_in_seNet = constrainted_se_grads_fromD

      ## D_GRL_008
      # shape_list = []
      # split_sizes = []
      # vec1 = tf.constant([])
      # vec2 = tf.constant([])
      # constrainted_se_grads_fromD = []
      # for grad1, grad2 in zip(se_loss_grads, d_grads_in_seNet):
      #   grad_shape = grad1.shape.as_list()
      #   shape_list.append(grad_shape)
      #   vec1_t = tf.reshape(grad1,[-1])
      #   vec2_t = tf.reshape(grad2,[-1])
      #   vec_len = vec1_t.shape.as_list()[0]
      #   split_sizes.append(vec_len)
      #   vec1 = tf.concat([vec1, vec1_t], 0)
      #   vec2 = tf.concat([vec2, vec2_t], 0)
      # prj_on_vec1 = tf.nn.relu(tf.reduce_sum(vec1*vec2,-1)/tf.reduce_sum(vec1*vec1, -1))*vec1
      # # print(len(shape_list), flush=True)
      # constrainted_se_grads_fromD = tf.split(prj_on_vec1, split_sizes)
      # constrainted_se_grads_fromD = [
      #     tf.reshape(grad, grad_shape) for grad, grad_shape in zip(constrainted_se_grads_fromD, shape_list)]
      # d_grads_in_seNet = constrainted_se_grads_fromD
    # endregion d_grad_in_seNet

    # region d_grads_in_D_net
    d_grads_in_D_Net = [grad*PARAM.discirminator_grad_coef for grad in d_grads_in_D_Net]
    # endregion d_grads_in_D_net

    all_grads = []
    all_params = []

    if "se_loss" in PARAM.D_used_losses:
      all_grads = se_loss_grads
      all_params = self.se_net_params

    if "deep_feature_loss" in PARAM.D_used_losses:
      if len(all_grads)==0:
        all_grads = deep_f_loss_grads
        all_params = self.se_net_params
      else:
        # merge se_grads from se_loss and deep_feature_loss
        all_grads = [grad1+grad2 for grad1, grad2 in zip(all_grads, deep_f_loss_grads)]

    assert len(all_grads)>0, "se_loss and deep_feature_loss are all turn off."

    if "D_loss" in PARAM.D_used_losses:
      if PARAM.se_grad_fromD_coef*ifD_passGrad_to_SE > 1e-12:
        # merge se_grads from D_loss
        all_grads = [grad1+grad2 for grad1, grad2 in zip(all_grads, d_grads_in_seNet)]

      if PARAM.discirminator_grad_coef > 1e-12:
        print('optimizer D')
        # merge d_grads_in_D_Net and D_params
        all_grads = all_grads + d_grads_in_D_Net
        all_params = self.se_net_params + self.d_params

    all_clipped_grads, _ = tf.clip_by_global_norm(all_grads, PARAM.max_gradient_norm)
    self._train_op = self.optimizer.apply_gradients(zip(all_clipped_grads, all_params),
                                                    global_step=self.global_step)


  def forward(self, mixed_wav_batch):
    r_outputs = self.real_networks_forward(mixed_wav_batch)
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = r_outputs

    return r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch

  def get_deep_features_losses(self, deep_features):
    '''
    deep_features: list,
    '''
    if PARAM.DFL_use_Dbottom_only:
      deep_features = deep_features[:1]
    losses = []
    for deep_f in deep_features[:-1]:
      labels, ests = tf.split(deep_f, 2, axis=0) # [batch,time,f]
      loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels-ests), axis=0))
      losses.append(loss)

    deep_f = deep_features[-1]
    if PARAM.deepFeatureLoss_softmaxLogits and not PARAM.simple_D: # simple_D not save logits as DeepFeatureLoss
      deep_f = tf.nn.softmax(deep_f)
    labels, ests = tf.split(deep_f, 2, axis=0) # [batch,time,f]
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels-ests), axis=0))
    losses.append(loss)
    n_deep = len(losses) * 1.0
    losses = [loss * PARAM.deepFeatureLoss_coef / n_deep for loss in losses]
    return losses


  def get_discriminator_loss(self, forward_outputs):
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = forward_outputs
    logits, one_hots_labels, deep_features = self.clean_and_enhanced_mag_discriminator(self.clean_mag_batch, r_est_clean_mag_batch)
    # print("23333333333333", one_hots_labels.shape.as_list(), logits.shape.as_list())
    loss = tf.losses.softmax_cross_entropy(one_hots_labels, logits) # max about 0.7
    deep_features_losses = self.get_deep_features_losses(deep_features)

    # w_DFL_ref_DLoss = 1.0/(1.0+tf.exp(tf.stop_gradient(loss + 1e-12)*40.0-4.0))
    w_DFL_ref_DLoss = tf.nn.sigmoid(4.0-tf.stop_gradient(loss)*PARAM.D_strict_degree_for_DFL) # loss+1e-12 is a new tf node
    if PARAM.weighted_DFL_by_DLoss:
      deep_features_losses = [dfloss*w_DFL_ref_DLoss for dfloss in deep_features_losses]

    loss = loss*PARAM.D_loss_coef
    return loss, deep_features_losses

  def get_loss(self, forward_outputs):
    clean_mag_batch_label = self.clean_mag_batch
    r_est_clean_mag_batch, r_est_clean_spec_batch, r_est_clean_wav_batch = forward_outputs

    if PARAM.add_logFilter_in_SE_Loss:
      a = self.variables._f_log_a
      b = self.variables._f_log_b
      c = self.variables._f_log_c
      clean_mag_batch_label = misc_utils.LogFilter_of_Loss(a,b,c,clean_mag_batch_label,
                                                           PARAM.LogFilter_type)
      r_est_clean_mag_batch = misc_utils.LogFilter_of_Loss(a,b,c,r_est_clean_mag_batch,
                                                           PARAM.LogFilter_type)

    # region real net losses
    ## frequency domain loss
    self.real_net_mag_mse = losses.batch_time_fea_real_mse(r_est_clean_mag_batch, clean_mag_batch_label)
    self.real_net_reMagMse = losses.batch_real_relativeMSE(r_est_clean_mag_batch, clean_mag_batch_label, PARAM.relative_loss_epsilon)
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

    loss = 0
    loss_names = PARAM.loss_name

    for i, name in enumerate(loss_names):
      name = name.replace('comp_net_', 'real_net_')
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
      }[name]
      if len(PARAM.loss_weight) > 0:
        loss_t *= PARAM.loss_weight[i]
      loss += loss_t
    return loss
