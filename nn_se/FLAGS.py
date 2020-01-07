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
  root_dir = '/home/lhf/worklhf/se-with-FTL/'
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
  DISCRIMINATOR_AD_MODEL
  """
  model_name = "DISCRIMINATOR_AD_MODEL"

  relative_loss_epsilon = 0.001
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sdrv3_bias = None # float, a bias will be added before vector dot multiply.
  stop_criterion_losses = None
  show_losses = None
  use_wav_as_feature = False
  net_out_mask = True
  frame_length = 256
  frame_step = 64
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  rnn_units = 256
  rlstmCell_implementation = 2
  fft_dot = 129
  max_keep_ckpt = 30
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 20
  batches_to_logging = 300

  max_model_abandon_time = 3
  no_abandon = True
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  # losses optimized in "DISCRIMINATOR_AD_MODEL"
  D_keep_prob = 0.8
  frame_level_D = True # discriminate frame is noisy or clean
  losses_position = ["not_transformed_losses", "transformed_losses"]
  FT_type = "LogValueT" # feature transformer type: "LogValueT", "FrequencyScaleT", "DenseT"

  """
  @param not_transformed_losses/transformed_losses[add FT before loss_name]:
  loss_mag_mse, loss_spec_mse, loss_wav_L1, loss_wav_L2,
  loss_reMagMse, loss_reSpecMse, loss_reWavL2,
  loss_sdrV1, loss_sdrV2, loss_stSDRV3, loss_cosSimV1, loss_cosSimV2,
  """
  not_transformed_losses = ["loss_mag_mse"]
  transformed_losses = ["FTloss_mag_mse"] # must based on magnitude spectrum
  NTloss_weight = []
  Tloss_weight = []
  stop_criterion_losses = ['loss_mag_mse']
  show_losses = ['loss_mag_mse', 'FTloss_mag_mse', 'd_loss']

  # just for "DISCRIMINATOR_AD_MODEL"
  discirminator_grad_coef = 1.0

  cnn_shortcut = None # None | "add" | "multiply"

  weighted_FTL_by_DLoss = False # if D_loss is large (about 0.7) w_FTL tends to 0.0, otherwise tends to 1.0
  D_strict_degree_for_FTL = 300.0 # for weighted_FTL_by_DLoss

  feature_type = "DFT" # DFT | DCT | QCT

  add_FeatureTrans_in_SE_inputs = False
  LogFilter_type = 2
  f_log_a = 1.0
  f_log_b = 0.0001
  log_filter_eps_a_b = 1e-6
  log_filter_eps_c = 0.001
  f_log_var_trainable = True

  use_noLabel_noisy_speech = False


class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/se-with-FTL'

class se_MagMSE(p40): # running p40
  '''
  baseline
  '''
  GPU_PARTION = 0.45
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_mag_mse']
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = 'LogValueT'
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_mag_mse']
  show_losses = ['loss_mag_mse', 'FTloss_mag_mse', 'd_loss']

class se_FTMagMSE_LogVT001(p40): # running p40
  '''
  LogVT
  '''
  GPU_PARTION = 0.45
  losses_position = ['transformed_losses']
  # not_transformed_losses = ['loss_mag_mse']
  transformed_losses = ['FTloss_mag_mse']
  FT_type = 'LogValueT'
  weighted_FTL_by_DLoss = False
  add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_mag_mse']
  show_losses = ['loss_mag_mse', 'FTloss_mag_mse', 'd_loss']

class se_FTMagMSE_LogVT001_complexD(p40): # running p40
  '''
  LogVT
  '''
  GPU_PARTION = 0.45
  losses_position = ['transformed_losses']
  # not_transformed_losses = ['loss_mag_mse']
  transformed_losses = ['FTloss_mag_mse']
  FT_type = 'LogValueT'
  weighted_FTL_by_DLoss = False
  add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_mag_mse']
  show_losses = ['loss_mag_mse', 'FTloss_mag_mse', 'd_loss']

PARAM = se_FTMagMSE_LogVT001_complexD

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
