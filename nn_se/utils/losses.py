import tensorflow as tf
import numpy as np

def batchMean_SSNR(est, ref):
  #est, ref: [N, S]
  eps = np.finfo(np.float64).eps
  noise = ref - est
  st_noi = tf.signal.frame(noise, frame_length=480, # [batch, frame, st_wav]
                           frame_step=120, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=480,
                           frame_step=120, pad_end=True)
  noi_temp = tf.reduce_sum(tf.square(st_noi), -1) #[N, T]
  ref_temp = tf.reduce_sum(tf.square(st_ref), -1)
  ssnr = 10*tf.log(ref_temp / noi_temp + eps) / tf.log(10.0)
  loss_ssnr = -tf.reduce_mean(ssnr)
  return loss_ssnr

def vec_dot_mul(y1, y2):
  dot_mul = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
  # print('dot', dot_mul.size())
  return dot_mul

def vec_normal(y):
  normal_ = tf.sqrt(tf.reduce_sum(tf.square(y), axis=-1))
  # print('norm',normal_.size())
  return normal_

def mag_fn(real, imag):
  return tf.sqrt(real**2+imag**2)

def FSum_MSE(y1, y2, _idx=2.0):
  # y1, y2: [N, T, F] real or complex
  if y1.dtype is tf.complex128 or y1.dtype is tf.complex64:
    return 0.5*(FSum_MSE(tf.real(y1), tf.real(y2)) + FSum_MSE(tf.imag(y1), tf.imag(y2)))

  loss = (y1-y2)**_idx
  loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
  return loss

def FSum_compressedMag_mse(y1, y2, compress_idx, eps=1e-5):
  """
  y1>=0: real, [batch, F, T]
  y2>=0: real, [batch, F, T]
  """
  y1 = tf.pow(y1+eps, compress_idx)
  y2 = tf.pow(y2+eps, compress_idx)
  loss = FSum_MSE(y1, y2)
  return loss

def FSum_compressedStft_mse(est_mag, est_normstft, clean_mag, clean_normstft, compress_idx, eps=1e-5):
  """
  est_mag:                real, [batch, T, F]
  est_normstft:   (real, imag),
  clean_mag:              real,
  clean_normstft: (real, imag),
  """
  # compress_idx = 1.0
  est_abs_cpr = tf.pow(est_mag+eps, compress_idx) # [batch, T, F]
  clean_abs_cpr = tf.pow(clean_mag+eps, compress_idx)

  est_cpr_stft = tf.complex(est_abs_cpr, 0.0) * est_normstft
  clean_cpr_stft = tf.complex(clean_abs_cpr, 0.0) * clean_normstft
  loss = FSum_MSE(est_cpr_stft, clean_cpr_stft)
  return loss

def FSum_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  # y1, y2: [N, T, F] real or complex
  if y1.dtype is tf.complex128 or y1.dtype is tf.complex64:
    return 0.5*(FSum_relativeMSE(
        tf.real(y1), tf.real(y2), RL_epsilon, index_) + FSum_relativeMSE(
        tf.imag(y1), tf.imag(y2), RL_epsilon, index_))
  relative_loss = tf.abs(y1-y2)/(tf.abs(y1)+tf.abs(y2)+RL_epsilon)
  loss = tf.pow(relative_loss, index_)
  loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
  return loss

def FSum_MAE(y1, y2):
  # y1, y2: [N, T, F] real or complex
  if y1.dtype is tf.complex128 or y1.dtype is tf.complex64:
    return 0.5*(FSum_MAE(tf.real(y1), tf.real(y2)) + FSum_MAE(tf.imag(y1), tf.imag(y2)))
  loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y1-y2), axis=-1))
  return loss

def FSum_relativeMAE(y1, y2, RL_epsilon):
  # y1, y2: [N, T, F] real or complex
  if y1.dtype is tf.complex128 or y1.dtype is tf.complex64:
    return 0.5*(FSum_relativeMAE(
        tf.real(y1), tf.real(y2), RL_epsilon) + FSum_relativeMAE(tf.imag(y1), tf.imag(y2), RL_epsilon))
  relative_loss = tf.abs(y1-y2)/(tf.abs(y1)+tf.abs(y2)+RL_epsilon)
  loss = tf.reduce_mean(tf.reduce_sum(relative_loss, axis=-1))
  return loss

def batchMean_CosSim_loss(est, ref): # -cos
  '''
  est, ref: [batch, ..., n_sample]
  '''
  # print(est.size(), ref.size(), flush=True)
  cos_sim = - tf.divide(vec_dot_mul(est, ref), # [batch, ...]
                        tf.multiply(vec_normal(est), vec_normal(ref)))
  loss = tf.reduce_mean(cos_sim)
  return loss

def batchMean_SquareCosSim_loss(est, ref): # -cos^2
  # print('23333')
  loss_s1 = - tf.divide(vec_dot_mul(est, ref)**2,  # [batch, ...]
                        tf.multiply(vec_dot_mul(est, est), vec_dot_mul(ref, ref)))
  loss = tf.reduce_mean(loss_s1)
  return loss

def batchMean_short_time_CosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss = batchMean_CosSim_loss(st_est, st_ref)
  return loss

def batchMean_short_time_SquareCosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos^2
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss = batchMean_SquareCosSim_loss(st_est, st_ref)
  return loss
