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

  u_times = 1.0
  u_eps = 1e-6
  f_u_var_trainable = True
  n_u_var = 128
  FT_type = ["trainableUlaw_v2"] # feature transformer type: "dense", "trainableUlaw_v2"
  add_ulawFT_in_G = False
  # MelDenseT_n_mel = 80
  # melDenseT_trainable = True
  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt


class p40(BaseConfig):
  root_dir = '/home/zhangwenbo5/lihongfeng/se-with-FTL'

###########################################
class se_magmse(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_mag_mse"]
  sum_losses_D = []
  show_losses = ["loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class se_magmse_ssnr(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_mag_mse", "loss_ssnr"]
  sum_losses_G_w = [1.0, 0.2]
  sum_losses_D = []
  show_losses = ["loss_mag_mse", "loss_ssnr",
                 "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class se_magmse_spreadN(p40): # done v100
  '''
   net_output*0.01
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_mag_mse"]
  sum_losses_D = []
  show_losses = ["d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class se_magremse(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_mag_reMse"]
  sum_losses_D = []
  show_losses = ["loss_mag_reMse",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []
  relative_loss_epsilon = 0.05

class se_magremse_ssnr(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_mag_reMse", "loss_ssnr"]
  sum_losses_D = []
  show_losses = ["loss_mag_reMse", "loss_ssnr",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []
  relative_loss_epsilon = 0.05


class se_wavL1(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_wav_L1"]
  sum_losses_D = []
  show_losses = ["loss_wav_L1",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []
  frame_length = 512
  frame_step = 128

class se_ulawwavL1(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_ulawwav_L1"]
  sum_losses_D = []
  show_losses = ["loss_ulawwav_L1",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class se_ssnr(p40): # done v100
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_ssnr"]
  sum_losses_D = []
  show_losses = ["loss_ssnr",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []


######################################################
class logmagmse(p40): # done
  '''
  G: loss_logmag_mse
  D:
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_logmag_mse"]
  sum_losses_D = []
  show_losses = ["loss_logmag_mse",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_ulawV2_G_logmagmse_Ndloss_001(p40): # done v100
  '''
  u-law v2 128 var
  G: 0.5*loss_logmag_mse, -1*d_loss
  D: d_loss
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["loss_logmag_mse", "d_loss"]
  sum_losses_G_w = [0.5, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["d_loss", "loss_logmag_mse",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

# class dse_dense_G_FTmagmse_Ndloss_001(p40): # done v100
#   '''
#   u-law v2 128 var
#   G: FTloss_mag_mse + -1*d_loss
#   D: d_loss
#   '''
#   FT_type = ["dense"]
#   sum_losses_G = ["FTloss_mag_mse", "d_loss"]
#   sum_losses_G_w = [1.0, -1.0]
#   sum_losses_D = ["d_loss"]
#   show_losses = ["FTloss_mag_mse", "d_loss",
#                  "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
#   stop_criterion_losses = []

class dse_logE_G_FTmagmse_Ndloss_001(BaseConfig): # running 15041
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + -1*d_loss
  D: d_loss
  logE FT
  '''
  FT_type = ["logE"]
  sum_losses_G = ["FTloss_mag_mse", "d_loss"]
  sum_losses_G_w = [1.0, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_tlogE_G_FTmagmse_Ndloss_001(BaseConfig): # running 15041
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + -1*d_loss
  D: d_loss
  trainable_LogE_FT
  '''
  FT_type = ["trainableLogE"]
  sum_losses_G = ["FTloss_mag_mse", "d_loss"]
  sum_losses_G_w = [1.0, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_255ulawV2_G_FTmagmse_Ndloss_001(BaseConfig): # running 15041
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + -1*d_loss
  D: d_loss
  fixed u=255
  '''
  FT_type = ["trainableUlaw_v2"]
  f_u_var_trainable = False
  u_eps = 255
  sum_losses_G = ["FTloss_mag_mse", "d_loss"]
  sum_losses_G_w = [1.0, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_ulawV2_G_FTmagmse_Ndloss_001(p40): # done v100
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + -1*d_loss
  D: d_loss
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "d_loss"]
  sum_losses_G_w = [1.0, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_ulawV2_G_FTmagmse_logmagmse_Ndloss_001(p40): # done v100
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + 0.2*logmagmse + -1*d_loss
  D: d_loss
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "loss_logmag_mse", "d_loss"]
  sum_losses_G_w = [1.0, 0.2, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "loss_logmag_mse", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

class dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001(p40): # done v100
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + 0.2*loss_ssnr + -1*d_loss
  D: d_loss
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "loss_ssnr", "d_loss"]
  sum_losses_G_w = [1.0, 0.2, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "loss_ssnr", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []



class dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna(p40): # done v100
  '''
  u-law v2 128 var
  G: FTloss_mag_mse + 0.2*loss_ssnr + -1*d_loss
  D: d_loss
  '''
  FT_type = ["trainableUlaw_v2"]
  sum_losses_G = ["FTloss_mag_mse", "loss_ssnr", "d_loss"]
  sum_losses_G_w = [1.0, 0.2, -1.0]
  sum_losses_D = ["d_loss"]
  show_losses = ["FTloss_mag_mse", "loss_ssnr", "d_loss",
                 "loss_mag_mse", "loss_stft_mse", "loss_CosSim"]
  stop_criterion_losses = []

  datasets_name = 'noisy_datasets_16k_specAna'
  n_train_set_records = 20
  max_step = 40310
  step_to_save = 100

PARAM = dse_255ulawV2_G_FTmagmse_Ndloss_001

# PARAM = dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001

# CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python -m dse_255ulawV2_G_FTmagmse_Ndloss_001._2_train
