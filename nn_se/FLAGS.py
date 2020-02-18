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
  datasets_name = 'noisy_datasets_16k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/enhanced_testsets: enhanced results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"


  sampling_rate = 16000
  fft_dot = 257
  frame_length = 512
  frame_step = 256
  rnn_units = 512

  train_noisy_set = 'noisy_trainset_wav'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav']
  test_clean_sets = ['clean_testset_wav']

  n_train_set_records = 11572
  n_val_set_records = 824
  n_test_set_records = 824

  train_val_wav_seconds = 3.0

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  batch_size = 64
  n_processor_tfdata = 4

  """
  @param model_name:
  DISCRIMINATOR_AD_MODEL
  """
  model_name = "DISCRIMINATOR_AD_MODEL"

  relative_loss_epsilon = 0.02
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sdrv3_bias = None # float, a bias will be added before vector dot multiply.
  stop_criterion_losses = None
  show_losses = None
  net_out_mask = True
  frame_length = 512
  frame_step = 256
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  rnn_units = 512
  rlstmCell_implementation = 2
  fft_dot = 257
  max_keep_ckpt = 30
  optimizer = "Adam" # "Adam" | "RMSProp"
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 120
  batches_to_logging = 100

  max_model_abandon_time = 3
  no_abandon = True
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 1000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  # losses optimized in "DISCRIMINATOR_AD_MODEL"
  D_keep_prob = 0.8
  frame_level_D = False # discriminate frame is noisy or clean
  losses_position = ["not_transformed_losses", "transformed_losses", "d_loss"]
  FT_type = ["LogValueT"] # feature transformer type: "LogValueT", "RandomDenseT", "MelDenseT"
  MelDenseT_n_mel = 80
  melDenseT_trainable = True
  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt

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
  add_noisy_class_in_D = False
  discirminator_grad_coef = 1.0
  D_loss_coef = 1.0

  cnn_shortcut = None # None | "add" | "multiply"

  feature_transformer_grad_coef = 1.0
  weighted_FTL_by_DLoss = False # if D_loss is large (about 0.7) w_FTL tends to 0.0, otherwise tends to 1.0
  D_strict_degree_for_FTL = 300.0 # for weighted_FTL_by_DLoss

  feature_type = "DFT" # DFT | DCT | QCT | WAV | ComplexDFT

  add_FeatureTrans_in_SE_inputs = False
  LogFilter_type = 3
  logFT_type2_btimes = 1.0
  f_log_a = 1.0 # smaller, curve max smaller
  f_log_b = 0.001 # smaller, curve straighter
  log_filter_eps_a_b = 1e-6
  f_log_var_trainable = True

  use_noLabel_noisy_speech = False

  inverse_Win_in_stft = True



class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/se-with-FTL'

class se_reMagMSE_0100(BaseConfig): # done 15123
  '''
  baseline noisy datasets
  '''
  GPU_PARTION = 0.46
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.1
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

class se_reMagMSE_moreData(p40): # running p40
  '''
  baseline noisy datasets
  '''
  GPU_PARTION = 0.23
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.1
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

class se_reMagMSE_longwav(p40): # running p40
  '''
  baseline noisy datasets
  '''
  GPU_PARTION = 0.23
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.1
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

  train_val_wav_seconds = 5.0

class se_reMagMSE_0050(BaseConfig): # done 15123
  '''
  baseline noisy datasets
  '''
  GPU_PARTION = 0.46
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.05
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

class se_reMagMSE_3blstm(BaseConfig): # running 15123
  '''
  3 layers blstm
  '''
  GPU_PARTION = 0.46
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.1
  blstm_layers = 3
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

class se_reMagMSE_cnn(BaseConfig): # running 15123
  '''
  add cnn
  '''
  GPU_PARTION = 0.46
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_reMagMse']
  relative_loss_epsilon = 0.1
  no_cnn = False
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # weighted_FTL_by_DLoss = False
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_reMagMse']
  show_losses = ['loss_reMagMse', 'FTloss_mag_mse', 'd_loss']

PARAM = se_reMagMSE_moreData

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
