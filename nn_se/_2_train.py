import tensorflow as tf
import os
import numpy as np
import time
import collections
from pathlib import Path
import sys

from .models import model_builder
from .models import modules
from .dataloader import dataloader
from .utils import misc_utils
from .inference import enhance_one_wav
from .inference import SMG
from .utils import audio
from .sepm import compare
from .FLAGS import PARAM

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("avg_sum_loss_G", "avg_sum_loss_D", "avg_show_losses", "cost_time"))):
  pass


def eval_one_epoch(sess, val_model, initializer):
  sess.run(initializer)
  val_s_time = time.time()
  ont_batch_time = time.time()

  i = 0
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  avg_sum_loss_G = None
  avg_sum_loss_D = None
  avg_show_losses = None
  while True:
    try:
      (sum_loss_G, sum_loss_D, show_losses,
       ) = sess.run([val_model.losses.sum_loss_G,
                     val_model.losses.sum_loss_D,
                     val_model.losses.show_losses,
                     ])
      if avg_sum_loss_G is None:
        avg_sum_loss_G = sum_loss_G
        avg_sum_loss_D = sum_loss_D
        avg_show_losses = show_losses
      else:
        avg_sum_loss_G += sum_loss_G
        avg_sum_loss_D += sum_loss_D
        avg_show_losses += show_losses
      i += 1
      # if i==5: break
      print("\r", end="")
      print("validate: %d/%d, cost %.2fs, sum_loss[G %.2f, D %.2f], show_losses %s"
            "          " % (
                i, total_i, time.time()-ont_batch_time, sum_loss_G, sum_loss_D,
                str(show_losses)
            ),
            flush=True, end="")
      ont_batch_time = time.time()
    except tf.errors.OutOfRangeError:
      break

  print("\r                                                                       "
        "                                                                       \r", end="")
  avg_sum_loss_G /= total_i
  avg_sum_loss_D /= total_i
  avg_show_losses /= total_i
  return EvalOutputs(avg_sum_loss_G=avg_sum_loss_G,
                     avg_sum_loss_D=avg_sum_loss_D,
                     avg_show_losses=avg_show_losses,
                     cost_time=time.time()-val_s_time)

class TestOutputs(
    collections.namedtuple("TestOutputs",
                           ("csig", "cbak", "covl", "pesq", "ssnr",
                            "cost_time"))):
  pass

def test_one_epoch(sess, test_model):
  t1 = time.time()
  smg = SMG(session=sess, model=test_model, graph=None)
  testset_name = PARAM.test_noisy_sets[0]
  testset_dir = misc_utils.datasets_dir().joinpath(testset_name)
  _dir = misc_utils.enhanced_testsets_save_dir(testset_name)
  if _dir.exists():
    import shutil
    shutil.rmtree(str(_dir))
  _dir.mkdir(parents=True)
  enhanced_save_dir = str(_dir)

  noisy_path_list = list(map(str, testset_dir.glob("*.wav")))
  noisy_num = len(noisy_path_list)
  for i, noisy_path in enumerate(noisy_path_list):
    print("\renhance test wavs: %d/%d" % (i, noisy_num), flush=True, end="")
    noisy_wav, sr = audio.read_audio(noisy_path)
    enhanced_wav = enhance_one_wav(smg, noisy_wav)
    noisy_name = Path(noisy_path).stem
    audio.write_audio(os.path.join(enhanced_save_dir, noisy_name+'_enhanced.wav'),
                      enhanced_wav, PARAM.sampling_rate)
  print("\r                                                               \r", end="", flush=True)

  testset_name, cleanset_name = PARAM.test_noisy_sets[0], PARAM.test_clean_sets[0]
  print("\rCalculate PM %s:" % testset_name, flush=True, end="")
  ref_dir = str(misc_utils.datasets_dir().joinpath(cleanset_name))
  deg_dir = str(misc_utils.enhanced_testsets_save_dir(testset_name))
  res = compare(ref_dir, deg_dir, False)

  pm = np.array([x[1:] for x in res])
  pm = np.mean(pm, axis=0)
  pm = tuple(pm)
  print("\r                                                                    "
        "                                                                      \r", end="")
  t2 = time.time()
  return TestOutputs(csig=pm[0], cbak=pm[1], covl=pm[2], pesq=pm[3], ssnr=pm[4],
                     cost_time=t2-t1)


def main():
  train_log_file = misc_utils.train_log_file_dir()
  ckpt_dir = misc_utils.ckpt_dir()
  hparam_file = misc_utils.hparams_file_dir()
  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  if not ckpt_dir.exists():
    os.mkdir(str(ckpt_dir))

  misc_utils.save_hparams(str(hparam_file))

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      noisy_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_noisy_set)
      clean_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_clean_set)
      noisy_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_noisy_set)
      clean_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_clean_set)
      train_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_trainset_wav,
                                                                        clean_trainset_wav)
      val_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_valset_wav,
                                                                      clean_valset_wav)
      test_noisy_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, None], name='mixed_batch')

    ModelC, G, D = model_builder.get_model_class_and_var()

    generator = G()
    discriminator = D()
    train_model = ModelC(PARAM.MODEL_TRAIN_KEY, generator, discriminator, train_inputs.mixed, train_inputs.clean)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(PARAM.MODEL_VALIDATE_KEY, generator, discriminator, val_inputs.mixed,val_inputs.clean)
    test_model = ModelC(PARAM.MODEL_INFER_KEY, generator, discriminator, test_noisy_ph)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    # misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  # config.gpu_options.per_process_gpu_memory_fraction = PARAM.GPU_PARTION
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  ckpts = tf.train.get_checkpoint_state(str(misc_utils.ckpt_dir()))
  if ckpts is not None:
    ckpt = ckpts.model_checkpoint_path
    train_model.saver.restore(sess, ckpt)
  else:
    # region validation before training
    misc_utils.print_log("\n\n", train_log_file)
    misc_utils.print_log("sum_losses_G: "+str(PARAM.sum_losses_G)+"\n", train_log_file)
    misc_utils.print_log("sum_losses_D: "+str(PARAM.sum_losses_D)+"\n", train_log_file)
    misc_utils.print_log("show losses: "+str(PARAM.show_losses)+"\n", train_log_file)
    evalOutputs_prev = eval_one_epoch(sess, val_model, val_inputs.initializer)
    val_msg = "PRERUN.val> sum_loss:[G %.4F, D %.4f], show_losses:%s, Time:%.2Fs.\n\n\n" % (
        evalOutputs_prev.avg_sum_loss_G,
        evalOutputs_prev.avg_sum_loss_D,
        evalOutputs_prev.avg_show_losses,
        evalOutputs_prev.cost_time)
    misc_utils.print_log(val_msg, train_log_file)
    # endregion

  avg_sum_loss_G = None
  avg_sum_loss_D = None
  avg_show_losses = None
  save_time = time.time()
  init_inputs_times = 1
  sess.run(train_inputs.initializer)
  while True:
    try:
      one_batch_time = time.time()
      (sum_loss_G, sum_loss_D, show_losses, _,
       lr, u
       ) = sess.run([train_model.losses.sum_loss_G,
                     train_model.losses.sum_loss_D,
                     train_model.losses.show_losses,
                     train_model.train_op,
                     train_model.lr,
                     train_model.discriminator._f_u,
                    #  train_model.adam_p[:2]
                     ])
      global_step = sess.run(train_model.global_step)
      # print(adam_p)
      if global_step > PARAM.max_step:
        sess.close()
        misc_utils.print_log("\n", train_log_file, no_time=True)
        msg = '################### Training Done. ###################\n'
        misc_utils.print_log(msg, train_log_file)
        print('initial inputs %d times' % init_inputs_times)
        break
      if avg_sum_loss_G is None:
        avg_sum_loss_G = sum_loss_G
        avg_sum_loss_D = sum_loss_D
        avg_show_losses = show_losses
      else:
        avg_sum_loss_G += sum_loss_G
        avg_sum_loss_D += sum_loss_D
        avg_show_losses += show_losses

      u_str = "#(u %.2e)" % u
      print("\rtrain step: %d/%d, cost %.2fs, sum_loss[G %.2f, D %.2f], show_losses %s, lr %.2e, %s          " % (
            global_step, PARAM.max_step, time.time()-one_batch_time, sum_loss_G, sum_loss_D,
            str(show_losses), lr, u_str),
            flush=True, end="")
      one_batch_time = time.time()

      if global_step % PARAM.step_to_save == 0:
        print("\r                                                                           "
              "                                                                             "
              "                                                                   \r", end="")
        avg_sum_loss_G /= PARAM.step_to_save
        avg_sum_loss_D /= PARAM.step_to_save
        avg_show_losses /= PARAM.step_to_save
        misc_utils.print_log("  Save step %03d:\n" % global_step, train_log_file)
        misc_utils.print_log("     sum_losses_G: "+str(PARAM.sum_losses_G)+"\n", train_log_file)
        misc_utils.print_log("     sum_losses_D: "+str(PARAM.sum_losses_D)+"\n", train_log_file)
        misc_utils.print_log("     show losses : "+str(PARAM.show_losses)+"\n", train_log_file)
        misc_utils.print_log("     Train     > sum_loss:[G %.4f, D %.4f], show_losses:%s, lr:%.2e, %s, Time:%ds.      \n" % (
            avg_sum_loss_G, avg_sum_loss_D, str(avg_show_losses), lr, u_str, time.time()-save_time),
            train_log_file)

        # val
        evalOutputs = eval_one_epoch(sess, val_model, val_inputs.initializer)
        misc_utils.print_log("     Validation> sum_loss:[G %.4f, D %.4f], show_losses:%s, Time:%ds.            \n" % (
            evalOutputs.avg_sum_loss_G, evalOutputs.avg_sum_loss_D,
            str(evalOutputs.avg_show_losses), evalOutputs.cost_time),
            train_log_file)

        # test
        testOutputs = test_one_epoch(sess, test_model)
        misc_utils.print_log("     Test      > Csig: %.3f, Cbak: %.3f, Covl: %.3f, pesq: %.3f,"
                             " ssnr: %.4f, Time:%ds.           \n" % (
                                 testOutputs.csig, testOutputs.cbak, testOutputs.covl, testOutputs.pesq,
                                 testOutputs.ssnr, testOutputs.cost_time),
                             train_log_file)

        # save ckpt
        ckpt_name = PARAM().config_name()+('_step%06d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
            global_step, avg_sum_loss_G, evalOutputs.avg_sum_loss_G, lr,
            time.time()-save_time))
        train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))

        misc_utils.print_log("\n", train_log_file, no_time=True)

        # init var
        avg_sum_loss_G = None
        avg_sum_loss_D = None
        avg_show_losses = None
        save_time = time.time()
    except tf.errors.OutOfRangeError:
      sess.run(train_inputs.initializer)
      init_inputs_times += 1


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
