from .dataloader.dataloader import get_batch_inputs_from_dataset
from .dataloader.dataloader import get_batch_inputs_from_nosiyCleanDataset
from .utils import audio
from .FLAGS import PARAM
import tensorflow as tf
import numpy as np
import os
from .utils import misc_utils

def test_dataloader_py():
  batch=get_batch_inputs_from_dataset(PARAM.train_name)
  sess=tf.compat.v1.Session()
  sess.run(batch.initializer)
  clean, noise, mixed=sess.run([batch.clean, batch.noise, batch.mixed])
  print(np.shape(clean))
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/clean.wav"),clean[0],PARAM.sampling_rate)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/noise.wav"),noise[0],PARAM.sampling_rate)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/mixed.wav"),mixed[0],PARAM.sampling_rate)


def wav_through_stft_istft():
  step = 64
  dataset_dir = misc_utils.datasets_dir()
  testdata_dir = dataset_dir.joinpath(PARAM.test_name)
  wav_dir = testdata_dir.joinpath("speech", "p265", "p265_002.wav")
  wav, sr = audio.read_audio(str(wav_dir))
  wav_batch = np.array([wav], dtype=np.float32)
  spec = misc_utils.tf_wav2feature(wav_batch, PARAM.frame_length, step)

  mag = tf.math.abs(spec)
  phase = tf.math.angle(spec)
  spec2 = tf.complex(mag, 0.0) * tf.exp(tf.complex(0.0, phase))

  wav2 = misc_utils.tf_feature2wav(spec2, PARAM.frame_length, step)

  sess = tf.compat.v1.Session()
  wav_np = sess.run(wav2)
  wav_np = wav_np[0]
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/p265_002_reconstructed_step%d.wav" % step),wav_np,PARAM.sampling_rate)


def wav_through_stft_istft_noreconstructed():
  dataset_dir = misc_utils.datasets_dir()
  testdata_dir = dataset_dir.joinpath(PARAM.test_name)
  wav_dir = testdata_dir.joinpath("speech", "p265", "p265_002.wav")
  wav, sr = audio.read_audio(str(wav_dir))
  wav_batch = np.array([wav], dtype=np.float32)
  spec = misc_utils.tf_wav2feature(wav_batch, PARAM.frame_length, PARAM.frame_step)

  # mag = tf.math.abs(spec)
  # phase = tf.math.angle(spec)
  # spec2 = tf.cast(mag, tf.dtypes.complex64) * tf.exp(1j*tf.cast(phase, tf.dtypes.complex64))

  spec2 = spec

  wav2 = misc_utils.tf_feature2wav(spec2, PARAM.frame_length, PARAM.frame_step)

  sess = tf.compat.v1.Session()
  wav_np = sess.run(wav2)
  wav_np = wav_np[0]
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/p265_002_only_stft.wav"),wav_np,PARAM.sampling_rate)


def test_dataloader_from_noisy_clean_datasets():
  batch=get_batch_inputs_from_nosiyCleanDataset(PARAM.train_noisy_path, PARAM.train_clean_path)
  sess=tf.compat.v1.Session()
  sess.run(batch.initializer)
  clean, mixed=sess.run([batch.clean, batch.mixed])
  print(np.shape(clean), np.shape(mixed))
  if not os.path.exists("exp/test"):
    os.makedirs("exp/test/")
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/clean.wav"),clean[0],PARAM.sampling_rate)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/mixed.wav"),mixed[0],PARAM.sampling_rate)


if __name__ == "__main__":
  # test_dataloader_py()
  # wav_through_stft_istft()
  # wav_through_stft_istft_noreconstructed()
  test_dataloader_from_noisy_clean_datasets()
