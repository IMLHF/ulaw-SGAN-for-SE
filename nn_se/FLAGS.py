class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'val'
  MODEL_INFER_KEY = 'infer'

  # dataset name
  train_name="train"
  validation_name="validation"
  test_name="test"

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/speech_enhancement_2019_exp/'
  # datasets_name = 'vctk_musan_datasets'
  datasets_name = 'datasets'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/test_records: test results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"

  # _1_preprocess param
  n_train_set_records = 72000
  n_val_set_records = 7200
  n_test_set_records = 3600
  train_val_snr = [-5, 15]
  train_val_wav_seconds = 3.0

  sampling_rate = 8000

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  batch_size = 64
  n_processor_tfdata = 4

  """
  @param model_name:
  CNN_RNN_FC_REAL_MASK_MODEL, DISCRIMINATOR_AD_MODEL, CCNN_CRNN_CFC_COMPLEX_MASK_MODEL,
  RC_HYBIRD_MODEL, RR_HYBIRD_MODEL
  """
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"

  """
  @param loss_name:
  real_net_mag_mse, real_net_spec_mse, real_net_wav_L1, real_net_wav_L2,
  real_net_reMagMse, real_net_reSpecMse, real_net_reWavL2,
  real_net_sdrV1, real_net_sdrV2, real_net_sdrV3, real_net_stSDRV3, real_net_cosSimV1, real_net_cosSimV1WT10, real_net_cosSimV2,
  real_net_specTCosSimV1, real_net_specFCosSimV1, real_net_specTFCosSimV1,
  real_net_last_blstm_fb_orthogonal,
  """
  relative_loss_epsilon = 0.001
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sdrv3_bias = None # float, a bias will be added before vector dot multiply.
  loss_name = ["real_net_mag_mse"]
  stop_criterion_losses = None
  show_losses = None
  loss_weight = []
  use_wav_as_feature = False
  net_out_mask = True
  frame_length = 256
  frame_step = 64
  no_cnn = False
  blstm_layers = 2
  lstm_layers = 0
  post_lstm_layers = 2
  post_complex_net_output = 'cmask' # cmask | cresidual | cspec
  complex_clip_mag = False
  complex_clip_label_mag = False
  complex_clip_mag_max = 1.5
  rnn_units = 256
  rlstmCell_implementation = 1
  clstmCell_implementation = 2
  fft_dot = 129
  max_keep_ckpt = 30
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 100
  batches_to_logging = 300

  max_model_abandon_time = 3
  no_abandon = False
  no_stop = False
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  # losses optimized in "DISCRIMINATOR_AD_MODEL"
  frame_level_D = False # discriminate frame is noisy or clean
  simple_D = False # if true, frame_level_D must be true.
  D_used_losses = ["se_loss","D_loss","deep_feature_loss"]
  DFL_use_Dbottom_only = False

  # just for "DISCRIMINATOR_AD_MODEL"
  D_GRL = False
  discirminator_grad_coef = 1.0
  D_Grad_DCC = False # DCC:Direction Consistent Constraints
  se_grad_fromD_coef = 1.0
  D_loss_coef = 1.0

  cnn_shortcut = None # None | "add" | "multiply"

  deepFeatureLoss_softmaxLogits = False
  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = False # if D_loss is large (about 0.7) w_DFL tends to 0.0, otherwise tends to $deepFeatureLoss_coef
  D_strict_degree_for_DFL = 40.0 # for weighted_DFL_by_DLoss

  feature_type = "DFT" # DFT | DCT | QCT

  # just for "DISCRIMINATOR_AD_MODEL"
  add_logFilter_in_Discrimitor = False # add log Value Filter to features of Discriminator.
  add_logFilter_in_SE_inputs = False
  add_logFilter_in_SE_Loss = False # add log Value Filter to SE loss. Log Filter params have no grad.
  LogFilter_type = 1 # 1:a=0.1,b=1.0,c=0.001; 2:a=0.4,b=1.0,c=0.001
  f_log_a = 0.1
  f_log_b = 1.0
  f_log_c = 0.001
  log_filter_eps_a_b = 1e-6
  log_filter_eps_c = 1e-6
  f_log_var_trainable = True


class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/speech_enhancement_2019_exp'

class debug(p40):
  blstm_layers = 1
  lstm_layers = 1
  no_cnn = False
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"

class nn_se_lr0001(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0001

class nn_se_lr0003(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0003

class nn_se_lr001(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.001

class nn_se_lr003(p40): # done p40
  """
  cnn2blstm
  """
  learning_rate = 0.003

class nn_se_only2blstm(BaseConfig): # done 15123
  """
  only2blstm
  """
  no_cnn = True

class nn_se_only1blstm(BaseConfig): # done 15123
  """
  only1blstm
  """
  no_cnn = True
  blstm_layers = 1

class nn_se_cnn1blstm(p40): # done p40
  """
  cnn1blstm
  """
  blstm_layers = 1

class nn_se_cnn1blstm1lstm(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1

class nn_se_cMagMSE(p40): # not run p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128

class nn_se_rSpecMSE(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]

class nn_se_rSpecMSE_fixISTFTWindow(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]

class nn_se_rSpecMSE_DCT(p40): # done p40
  """
  cnn1blstm1lstm DCT features
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  feature_type = "DCT"
  fft_dot = 256

class nn_se_rSpecMSE_mulCnn(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  cnn_shortcut = "multiply"

class nn_se_rSpecMSE_addCnn(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  cnn_shortcut = "add"

class nn_se_rSpecMSE_D_GRL_001(p40): # done p40
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.5
  discirminator_grad_coef = 1.5

class nn_se_rSpecMSE_D_GRL_002(p40): # done p40
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.5

class nn_se_rSpecMSE_D_GRL_003(p40): # done p40
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 3.0

class nn_se_rSpecMSE_D_GRL_004(p40): # done p40
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  # loss_weight = [0.5]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_004_FLD(p40): # done p40
  '''
  FLD: frame level D
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  # loss_weight = [0.5]
  GPU_PARTION = 0.46
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0
  frame_level_D = True
  show_losses = ["real_net_spec_mse", "d_loss"]

class nn_se_rSpecMSE_D_GRL_005(p40): # done p40
  '''
  vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_006(p40): # done p40
  '''
  sign constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_306(p40): # done p40
  '''
  sign constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.1
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_406(p40): # done p40
  '''
  sign constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.2
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_007(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_307(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.1
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_007T1(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_307T1(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.1
  discirminator_grad_coef = 1.0
  show_losses = ["real_net_spec_mse", "d_loss"]


class nn_se_rSpecMSE_joint_SE_D_DFL(p40): # done p40 # dead
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]
  deepFeatureLoss_coef = 0.1

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 40.0

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD_S100(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 100.0

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD_S200(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 200.0

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD_S300(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD_S400(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 400.0

class nn_se_rSpecMSE_joint_SE_D_WDFLbyD_S500(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 500.0

class nn_se_rSpecMSE_joint_SE_simpleD_WDFLbyD_S300(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

class nn_se_rSpecMSE_joint_SE_simpleD2_WDFLbyD_S300(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

class nn_se_rSpecMSE_joint_SE_simpleD2_WDFLbyD_S300_noAbandon(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

  max_epoch = 20
  no_abandon = True

class nn_se_rSpecMSE_joint_SE_simpleD2_WDFLbyDbottom_S300(BaseConfig): # running 15123
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

  DFL_use_Dbottom_only = True

class nn_se_rSpecMSE_joint_SE_simpleD2_WDFLbyDbottom_S300_noAbandon(p40): # running p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

  DFL_use_Dbottom_only = True

  max_epoch = 20
  no_abandon = True

class nn_se_rSpecMSE_joint_SE_simpleD_WDFLbyD_S500(BaseConfig): # done 15123
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.99
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  # deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["se_loss", "deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 500.0

  simple_D = True

class nn_se_rSpecMSE_joint_simpleD2_WDFLbyD_S300(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "deep_features_loss", "d_loss", "deep_features_losses"]
  deepFeatureLoss_softmaxLogits = True
  D_used_losses = ["deep_feature_loss", "D_loss"]

  deepFeatureLoss_coef = 1.0
  weighted_DFL_by_DLoss = True
  D_strict_degree_for_DFL = 300.0

  frame_level_D = True
  simple_D = True

  max_epoch = 20
  no_abandon = True

class nn_se_rSpecMSE_joint_SE_only_noAbandon(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_spec_mse", "d_loss"]
  D_used_losses = ["se_loss"]

  simple_D = True

  max_epoch = 30
  no_abandon = True


class nn_se_rSpecMSE_D_GRL_008(p40): # done p40
  '''
  full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 1.0
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_GRL_308(p40): # done p40
  '''
  full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = True
  D_Grad_DCC = True
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3
  se_grad_fromD_coef = 0.1
  discirminator_grad_coef = 1.0

class nn_se_rSpecMSE_D_noGRL(p40): # done p40
  model_name = 'DISCRIMINATOR_AD_MODEL'
  D_GRL = False
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.3

class nn_se_rSpecMSEBlstmOrth(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_weight = [1.0, 1.0]
  loss_name = ["real_net_spec_mse", "real_net_last_blstm_fb_orthogonal"]

class nn_se_rSpecMSE_lstmv2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  rlstmCell_implementation = 2

class nn_se_hybirdSpecMSE_001(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cmask'
  GPU_PARTION = 0.32 #

class nn_se_hybirdSpecMSE_002(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.32 #

class nn_se_hybirdSpecMSE_003(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cspec'
  GPU_PARTION = 0.32 #

class nn_se_hybirdSpecMSE_004(p40): # stop p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 0.5]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.32 #

class nn_se_hybirdSpecMSE_005(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [0.5, 1.0]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.32 #

class nn_se_hybirdSpecMSEclipMag_001(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  clip post complex lstm inputs mag
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cmask'
  GPU_PARTION = 0.32 #
  complex_clip_mag = True
  complex_clip_mag_max = 1.5

class nn_se_hybirdSpecMSEclipMag_002(p40): # done p40
  """
  cnn1blstm1lstm+2bclstm1cdnn
  clip post complex lstm inputs mag
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.32 #
  complex_clip_mag = True
  complex_clip_mag_max = 1.5
  rlstmCell_implementation = 1
  clstmCell_implementation = 2

class nn_se_hybirdSpecMSEclipMag_003(p40): # done p40
  """
  cnn1blstm1lstm+2clstm1cdnn
  clip post complex lstm inputs mag
  """
  model_name = "RC_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.32 #
  complex_clip_mag = True
  complex_clip_label_mag = True
  complex_clip_mag_max = 1.5
  rlstmCell_implementation = 1
  clstmCell_implementation = 2

class nn_se_RRhybirdSpecMSE_001(p40): # done p40
  # cnn1blstm1lstm+2blstm1dnn
  model_name = "RR_HYBIRD_MODEL"
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", 'comp_net_spec_mse']
  loss_weight = [1.0, 1.0]
  post_complex_net_output = 'cresidual'
  GPU_PARTION = 0.30 #
  complex_clip_mag = True
  complex_clip_label_mag = True
  complex_clip_mag_max = 1.5
  rlstmCell_implementation = 1
  clstmCell_implementation = 2

class nn_se_cSpecMSE_doubleRnnVar(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  loss_name = ["real_net_spec_mse"]
  GPU_PARTION = 0.5
  # clstmCell_implementation = 2

class nn_se_cSpecMSE(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  # clstmCell_implementation = 2

class nn_se_cSpecMSE_clipMag(p40): # done p40
  """
  clip mag < 1.5
  """
  complex_clip_mag = True
  complex_clip_mag_max = 1.5
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]

class nn_se_cSpecMSE_real05(p40): # done p40
  """
  in complexLSTMCell, i and f multiply 0.5 not 0.5+0j
  """
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  # clstmCell_implementation = 2

class nn_se_cSpecMSE_cRNNno05(p40): # done p40
  """
  in complexLSTMCell, i and f multiply 0.7.
  """
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  # clstmCell_implementation = 2

class nn_se_cSpecMSE_lstmv2(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2

class nn_se_cSpecMSE_lstmv2_big(p40): # done p40
  blstm_layers = 1
  lstm_layers = 2
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 512
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2
  # GPU_PARTION = 0.7

class nn_se_cSpecMSE_lstmv2_lr01(p40): # stop p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2
  learning_rate = 0.01

class nn_se_cSpecMSE_lstmv2_lr003(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2
  learning_rate = 0.003

class nn_se_cSpecMSE_lstmv2_lr0001(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2
  learning_rate = 0.0001

class nn_se_cSpecMSE_lstmv2_nocnn(p40): # done p40
  blstm_layers = 1
  lstm_layers = 1
  model_name = "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL"
  rnn_units = 128
  loss_name = ["real_net_spec_mse"]
  clstmCell_implementation = 2
  no_cnn = True

class nn_se_rSpecMSE_noStop(BaseConfig): # stop 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  no_stop = True

class nn_se_rReMagMSE20(p40): # runnnig p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/20.0

class nn_se_rReMagMSE50(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/50.0

class nn_se_rReMagMSE100(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/100.0

class nn_se_rReMagMSE200(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/200.0

class nn_se_rReMagMSE500(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/500.0

class nn_se_rReMagMSE1000(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_epsilon = 1.0/1000.0

class nn_se_rReSpecMSE20(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/20.0

class nn_se_rReSpecMSE50(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/50.0

class nn_se_rReSpecMSE100(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/100.0

class nn_se_rReSpecMSE200(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/200.0

class nn_se_rReSpecMSE500(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/500.0

class nn_se_rReSpecMSE1000(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_epsilon = 1.0/1000.0

class nn_se_rWavL1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L1"]

class nn_se_rWavL2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L2"]

class nn_se_rReWavL2_AFD20(p40): # done p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 20
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_epsilon = 1.0/20.0

class nn_se_rReWavL2_AFD50(p40): # done p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 50
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_epsilon = 1.0/50.0

class nn_se_rReWavL2_AFD100(p40): # done p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 100
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_epsilon = 1.0/100.0

class nn_se_rReWavL2_AFD200(p40): # done p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 200
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_epsilon = 1.0/200.0

class nn_se_rReWavL2_AFD1000(p40): # done p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 1000
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_epsilon = 1.0/1000.0

class nn_se_rSDRv1(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV1"]

class nn_se_rSDRv2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV2"]

class nn_se_rSDRv3(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b005(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.05
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b010(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.1
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b050(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.5
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b100(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 1.0
  loss_name = ["real_net_sdrV3"]

class nn_se_rStSDRV3(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  short time SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_stSDRV3"]
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128

class nn_se_rStSDRV3_f1024(p40): # done p40
  """
  cnn1blstm1lstm
  short time SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_stSDRV3"]
  st_frame_length_for_loss = 1024
  st_frame_step_for_loss = 256

class nn_se_rCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV1"]

class nn_se_rCosSimV1WT10(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV1WT10"]

class nn_se_rCosSimV2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV2"]

class nn_se_rSpecTCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specTCosSimV1"]

class nn_se_rSpecFCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specFCosSimV1"]

class nn_se_rSpecTFCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specTFCosSimV1"]

class nn_se_rSpecMseSDRv3_1_1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0]

class nn_se_rMagSpecMseSDRv3_001(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0, 1.0]

class nn_se_rMagMseSDRv3_001(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0]

class nn_se_rMagSpecMse_001(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse"]
  loss_weight = [1.0, 1.0]

class nn_se_rWavL2SDRv3_1_1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L2", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0]

class nn_se_rSTWavMSE256(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]

class nn_se_rSTWavMSE512(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 512
  frame_step = 128
  fft_dot = 512
  loss_weight = [100.0]
  GPU_PARTION = 0.3

class nn_se_rSTWavMSE256Map(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]
  net_out_mask = False

class nn_se_rSTWavMSE256Map_noStop(BaseConfig): # stop 15123
  """
  cnn1blstm1lstm
  short time wav as feature
  no stop
  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]
  net_out_mask = False
  no_stop = True

class nn_se_rSTWavMSE512Map(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature
  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 512
  frame_step = 128
  fft_dot = 512
  loss_weight = [100.0]
  net_out_mask = False
  GPU_PARTION = 0.3

class nn_se_rMagMSE_AD_LogFilterLoss001(BaseConfig): # done 15041 # dead
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse"]
  GPU_PARTION = 0.95
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 1
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "d_loss"]

class nn_se_rMagMSE_AD_LogFilterLoss002(BaseConfig): # done 15043 #dead
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse"]
  GPU_PARTION = 0.90
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 2
  f_log_a = 0.4
  log_filter_eps_c = 0.001
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "d_loss"]

class nn_se_rMagMSE_AD_LogFilterLoss003(p40): # done p40 #DEAD
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse"]
  GPU_PARTION = 0.47
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0

  add_logFilter_in_Discrimitor = True # just on real_net_mag_mse and real_net_reMagMse
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 3
  log_filter_eps_c = 1e-12
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "d_loss"]

class nn_se_rMagMSE_AD_LogFilterLoss002a(p40): # done p40 # dead
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse"]
  GPU_PARTION = 0.45
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 2
  f_log_a = 0.4
  log_filter_eps_c = 0.001
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "real_net_spec_mse", "d_loss"]

class nn_se_rMagMSE_AD_LogFilterLoss002a_noAbandon(p40): # done p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse"]
  GPU_PARTION = 0.45
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 2
  f_log_a = 0.4
  log_filter_eps_c = 0.001
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "real_net_spec_mse", "d_loss"]
  no_abandon = True
  max_epoch = 30

class nn_se_rMagMSE_AD_LogFilterLoss002b_noAbandon(p40): # running p40
  '''
  half full vec constrained
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse"]
  GPU_PARTION = 0.45
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 2
  f_log_a = 1.0
  f_log_b = 0.0001
  f_log_c = 0.0
  log_filter_eps_c = 0.001
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "real_net_spec_mse", "d_loss"]
  no_abandon = True
  max_epoch = 20

#nn_se_rMagMSE_AD_LogFilterLoss002a_noAbandonLI :LI log inputs
class nn_se_rMagMSE_AD_LogFilterLoss002a_noAbandonLI(BaseConfig): # running 15043
  '''
  half full vec constrained
  add logFilter to se inputs feature
  '''
  model_name = 'DISCRIMINATOR_AD_MODEL'
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_mag_mse", "real_net_spec_mse"]
  GPU_PARTION = 0.99
  se_grad_fromD_coef = 0.0
  discirminator_grad_coef = 1.0
  add_logFilter_in_Discrimitor = True
  add_logFilter_in_SE_Loss = True
  LogFilter_type = 2
  f_log_a = 0.4
  f_log_c = 0.0
  log_filter_eps_c = 0.001
  stop_criterion_losses = ["real_net_spec_mse"]
  show_losses = ["real_net_mag_mse", "real_net_spec_mse", "d_loss"]
  no_abandon = True
  max_epoch = 20

  add_logFilter_in_SE_inputs = True

PARAM = nn_se_rSpecMSE_joint_SE_simpleD2_WDFLbyDbottom_S300_noAbandon

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
