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

  batch_size = 12
  learning_rate = 0.001

  """
  @param model_name:
  DISCRIMINATOR_AD_MODEL
  """
  model_name = "DISCRIMINATOR_AD_MODEL"

  relative_loss_epsilon = 0.1
  RL_idx = 2.0
  st_frame_length_for_loss = 400
  st_frame_step_for_loss = 160
  net_out_mask = True
  sampling_rate = 16000
  frame_length = 400
  frame_step = 160
  fft_length = 512
  blstm_layers = 3
  lstm_layers = 0
  D_blstm_layers = 2
  D_lstm_layers = 0
  blstm_drop_rate = 0.3
  rnn_units = 512
  rlstmCell_implementation = 2
  feature_dim = 257
  optimizer = "Adam" # "Adam" | "RMSProp"
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  max_step = 40010
  step_to_save = 1000

  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)

  # losses optimized in "DISCRIMINATOR_AD_MODEL"
  frame_level_D = False # discriminate frame is noisy or clean

  stft_norm_method = "polar" # polar | div

  """
  @param losses
  """
  sum_losses_G = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  sum_losses_G_w = []
  sum_losses_D = ["d_loss", "FTloss_mag_mse"]
  sum_losses_D_w = [1.0, -1.0]
  show_losses = ["loss_compressedMag_mse", "loss_compressedStft_mse"]
  show_losses_w = []
  stop_criterion_losses = []
  stop_criterion_losses_w = []
  loss_compressedMag_idx = 0.3

  # just for "DISCRIMINATOR_AD_MODEL"
  add_noisy_class_in_D = False

  u_times = 1.0
  u_eps = 1e-6
  f_u_var_trainable = True
  n_u_var = 128
  FT_type = [] # feature transformer type: "trainableUlaw", "trainableUlaw_v2"
  # MelDenseT_n_mel = 80
  # melDenseT_trainable = True
  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt


class p40(BaseConfig):
  root_dir = '/home/zhangwenbo5/lihongfeng/se-with-FTL'

###########################################

class dse_ulawV2(p40): # running v100
  '''
  u-law v2 128 var
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse"]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []
  # blstm_drop_rate = ?

class dse_ulawV2_stftMSE10(p40): # running v100
  '''
  u-law v2 128 var + stftMSE*1.0
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "loss_stft_mse"]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_ulawV2_stftMSE05(p40): # running v100
  '''
  u-law v2 128 var + stftMSE*1.0
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "loss_stft_mse"]
  sum_losses_G_w = [1.0, 0.5]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

PARAM = ulawV2var1024_stftMSE_divnorm

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m ulawV2var1024_stftMSE_divnorm._2_train
