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
from .FLAGS import PARAM


def __relative_impr(prev_, new_, declining=False):
  if declining:
    return (prev_-new_)/(abs(prev_)+abs(new_)+1e-8)
  return (new_-prev_)/(abs(prev_)+abs(new_)+1e-8)


class TrainOutputs(
    collections.namedtuple("TrainOutputs",
                           ("avg_loss", "avg_show_losses", "cost_time", "lr"))):
  pass


def train_one_epoch(sess, train_model, train_log_file,
                    stop_criterion_losses, show_losses):
  tr_loss, i, lr = 0.0, 0, -1
  total_show_losses_vec = None
  s_time = time.time()
  minbatch_time = time.time()
  one_batch_time = time.time()

  total_i = PARAM.n_train_set_records//PARAM.batch_size
  all_losses = {
    'loss_mag_mse': train_model.loss_mag_mse,
    'loss_reMagMse': train_model.loss_reMagMse,
    'loss_spec_mse': train_model.loss_spec_mse,
    'loss_reSpecMse': train_model.loss_reSpecMse,
    'loss_wav_L1': train_model.loss_wav_L1,
    'loss_wav_L2': train_model.loss_wav_L2,
    'loss_reWavL2': train_model.loss_reWavL2,
    'loss_sdrV1': train_model.loss_sdrV1,
    'loss_sdrV2': train_model.loss_sdrV2,
    'loss_cosSimV1': train_model.loss_cosSimV1,
    'loss_cosSimV2': train_model.loss_cosSimV2,
    'loss_stSDRV3': train_model.loss_stSDRV3,
    'd_loss': train_model.d_loss,
    'FTloss_mag_mse': train_model.FTloss_mag_mse,
    'FTloss_mag_RL': train_model.FTloss_mag_RL,
  }
  # show_losses = PARAM.show_losses if PARAM.show_losses is not None else PARAM.loss_name
  # stop_criterion_losses = PARAM.stop_criterion_losses if PARAM.stop_criterion_losses is not None else PARAM.loss_name
  losses_to_run = [train_model.train_op, train_model.lr, train_model.global_step]
  for l_name in show_losses:
    losses_to_run.append(all_losses[l_name])
  for l_name in stop_criterion_losses:
    losses_to_run.append(all_losses[l_name])

  # debug
  if PARAM.FT_type == "LogValueT" and "transformed_losses" in PARAM.losses_position:
    losses_to_run.extend([
        train_model.variables._f_log_a,
        train_model.variables._f_log_b,
    ])

  while True:
    try:
      run_out_losses = sess.run(losses_to_run)

      if PARAM.FT_type == "LogValueT" and "transformed_losses" in PARAM.losses_position:
        a,b = run_out_losses[-2:] # debug
        run_out_losses = run_out_losses[:-2]

      _, lr, global_step = run_out_losses[:3]
      runOut_show_losses = run_out_losses[3:len(show_losses)+3]
      runOut_show_losses = round_lists(runOut_show_losses, 4)
      runOut_show_losses_vec = np.array(unfold_list(runOut_show_losses))
      if total_show_losses_vec is None:
        total_show_losses_vec = np.zeros_like(runOut_show_losses_vec)
      total_show_losses_vec += runOut_show_losses_vec

      runOut_losses_stopCriterion = run_out_losses[-len(stop_criterion_losses):]
      sum_loss_stopCriterion = np.sum(runOut_losses_stopCriterion)

      tr_loss += sum_loss_stopCriterion
      i += 1
      print("\r", end="")
      abc = "#(a %.2e b %.2e)" % (
          a, b) if PARAM.FT_type == "LogValueT" and "transformed_losses" in PARAM.losses_position else "          "
      print("train: %d/%d, cost %.2fs, stop_loss %.2f, single_losses %s %s" % (
            i, total_i, time.time()-one_batch_time, sum_loss_stopCriterion,
            str(runOut_show_losses), abc),
            flush=True, end="")
      one_batch_time = time.time()
      if i % PARAM.batches_to_logging == 0:
        print("\r", end="")
        msg = "     Minbatch %04d: stop_loss:%.4f, losses:%s, lr:%.2e, time:%ds. %s\n" % (
                i, tr_loss/i, round_lists(list(total_show_losses_vec / i), 4), lr, time.time()-minbatch_time, abc
              )
        minbatch_time = time.time()
        misc_utils.print_log(msg, train_log_file)
    except tf.errors.OutOfRangeError:
      break
  print("\r", end="")
  e_time = time.time()
  tr_loss /= i
  avg_show_losses_vec = total_show_losses_vec / i
  return TrainOutputs(avg_loss=tr_loss,
                      avg_show_losses=round_lists(list(avg_show_losses_vec), 4),
                      cost_time=e_time-s_time,
                      lr=lr)


class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("avg_loss", "avg_show_losses", "cost_time"))):
  pass

def round_lists(lst, rd):
  return [round(n,rd) if type(n) is not list else round_lists(n,rd) for n in lst]

def unfold_list(lst):
  ans_lst = []
  [ans_lst.append(n) if type(n) is not list else ans_lst.extend(unfold_list(n)) for n in lst]
  return ans_lst

def eval_one_epoch(sess, val_model, stop_criterion_losses, show_losses):
  val_s_time = time.time()
  total_loss = 0.0
  ont_batch_time = time.time()

  i = 0
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  all_losses = {
    'loss_mag_mse': val_model.loss_mag_mse,
    'loss_reMagMse': val_model.loss_reMagMse,
    'loss_spec_mse': val_model.loss_spec_mse,
    'loss_reSpecMse': val_model.loss_reSpecMse,
    'loss_wav_L1': val_model.loss_wav_L1,
    'loss_wav_L2': val_model.loss_wav_L2,
    'loss_reWavL2': val_model.loss_reWavL2,
    'loss_sdrV1': val_model.loss_sdrV1,
    'loss_sdrV2': val_model.loss_sdrV2,
    'loss_cosSimV1': val_model.loss_cosSimV1,
    'loss_cosSimV2': val_model.loss_cosSimV2,
    'loss_stSDRV3': val_model.loss_stSDRV3,
    'd_loss': val_model.d_loss,
    'FTloss_mag_mse': val_model.FTloss_mag_mse,
    'FTloss_mag_RL': val_model.FTloss_mag_RL,
  }
  # show_losses = PARAM.show_losses if PARAM.show_losses is not None else PARAM.loss_name
  # stop_criterion_losses = PARAM.stop_criterion_losses if PARAM.stop_criterion_losses is not None else PARAM.loss_name
  losses_to_run = []
  for l_name in show_losses:
    losses_to_run.append(all_losses[l_name])
  for l_name in stop_criterion_losses:
    losses_to_run.append(all_losses[l_name])

  # losses_to_run.append(val_model.clean_mag_batch)
  # losses_to_run.append(val_model._debufgradients)

  total_show_losses_vec = None
  while True:
    try:
      run_out_losses = sess.run(losses_to_run)

      # debug_mag = run_out_losses[-1]
      # run_out_losses = run_out_losses[:-1]
      # print(debug_mag)

      runOut_show_losses = run_out_losses[:len(show_losses)]
      runOut_show_losses = round_lists(runOut_show_losses, 4)
      runOut_show_losses_vec = np.array(unfold_list(runOut_show_losses))
      if total_show_losses_vec is None:
        total_show_losses_vec = np.zeros_like(runOut_show_losses_vec)
      total_show_losses_vec += runOut_show_losses_vec

      runOut_losses_stopCriterion = run_out_losses[-len(stop_criterion_losses):]
      sum_loss_stopCriterion = np.sum(runOut_losses_stopCriterion)
      # print("\n", loss, loss_mag_mse, loss_spec_mse, losswavL1, losswavL2, flush=True)
      # print(np.mean(debug_mag), np.std(debug_mag), np.min(debug_mag), np.max(debug_mag), flush=True)
      total_loss += sum_loss_stopCriterion
      i += 1
      print("\r", end="")
      print("validate: %d/%d, cost %.2fs, stop_loss %.2f, single_losses %s"
            "          " % (
                i, total_i, time.time()-ont_batch_time, sum_loss_stopCriterion,
                str(runOut_show_losses)
            ),
            flush=True, end="")
      ont_batch_time = time.time()
    except tf.errors.OutOfRangeError:
      break

  print("\r", end="")
  avg_loss = total_loss / i
  avg_show_losses_vec = total_show_losses_vec / i
  val_e_time = time.time()
  return EvalOutputs(avg_loss=avg_loss,
                     avg_show_losses=round_lists(list(avg_show_losses_vec),4),
                     cost_time=val_e_time-val_s_time)


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
      train_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.train_name)
      val_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.validation_name)

    ModelC, VariablesC = model_builder.get_model_class_and_var()

    variables = VariablesC()
    train_model = ModelC(PARAM.MODEL_TRAIN_KEY, variables, train_inputs.mixed, train_inputs.clean)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(PARAM.MODEL_VALIDATE_KEY, variables, val_inputs.mixed,val_inputs.clean)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  # config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  config.gpu_options.per_process_gpu_memory_fraction = PARAM.GPU_PARTION
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  # region validation before training
  sess.run(val_inputs.initializer)
  stop_criterion_losses = PARAM.stop_criterion_losses if PARAM.stop_criterion_losses is not None else PARAM.loss_name
  show_losses = PARAM.show_losses if PARAM.show_losses is not None else PARAM.loss_name
  misc_utils.print_log("stop criterion losses: "+str(stop_criterion_losses)+"\n", train_log_file)
  misc_utils.print_log("show losses: "+str(show_losses)+"\n", train_log_file)
  evalOutputs_prev = eval_one_epoch(sess, val_model, stop_criterion_losses, show_losses)
  misc_utils.print_log("                                            "
                       "                                            "
                       "                                         \n\n",
                       train_log_file, no_time=True)
  val_msg = "PRERUN.val> StopLOSS:%.4F, ShowLOSS:%s, Cost itme:%.4Fs.\n" % (
      evalOutputs_prev.avg_loss,
      evalOutputs_prev.avg_show_losses,
      evalOutputs_prev.cost_time)
  misc_utils.print_log(val_msg, train_log_file)

  assert PARAM.s_epoch > 0, 'start epoch > 0 is required.'
  model_abandon_time = 0

  for epoch in range(PARAM.s_epoch, PARAM.max_epoch+1):
    misc_utils.print_log("\n\n", train_log_file, no_time=True)
    misc_utils.print_log("  Epoch %03d:\n" % epoch, train_log_file)
    misc_utils.print_log("   stop criterion losses: "+str(stop_criterion_losses)+"\n", train_log_file)
    misc_utils.print_log("   show losses: "+str(show_losses)+"\n", train_log_file)

    # train
    sess.run(train_inputs.initializer)
    trainOutputs = train_one_epoch(sess, train_model, train_log_file, stop_criterion_losses, show_losses)
    misc_utils.print_log("     Train     > stop_loss:%.4f, losses:%s, Cost time:%ds.             \n" % (
        trainOutputs.avg_loss,
        trainOutputs.avg_show_losses,
        trainOutputs.cost_time),
        train_log_file)

    # validation
    sess.run(val_inputs.initializer)
    evalOutputs = eval_one_epoch(sess, val_model, stop_criterion_losses, show_losses)
    val_loss_rel_impr = __relative_impr(evalOutputs_prev.avg_loss, evalOutputs.avg_loss, True)
    misc_utils.print_log("     Validation> stop_loss:%.4f, losses:%s, Cost time:%ds.             \n" % (
        evalOutputs.avg_loss,
        evalOutputs.avg_show_losses,
        evalOutputs.cost_time),
        train_log_file)

    # save or abandon ckpt
    ckpt_name = PARAM().config_name()+('_iter%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
        epoch, trainOutputs.avg_loss, evalOutputs.avg_loss, trainOutputs.lr,
        trainOutputs.cost_time+evalOutputs.cost_time))
    if val_loss_rel_impr > 0 or PARAM.no_abandon:
      train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))
      evalOutputs_prev = evalOutputs
      best_ckpt_name = ckpt_name
      msg = "     ckpt(%s) saved.\n" % ckpt_name
    else:
      model_abandon_time += 1
      # tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
      train_model.saver.restore(sess,
                                str(ckpt_dir.joinpath(best_ckpt_name)))
      # tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
      msg = "     ckpt(%s) abandoned.\n" % ckpt_name
    misc_utils.print_log(msg, train_log_file)

    # start lr halving
    if val_loss_rel_impr < PARAM.start_halving_impr and (not PARAM.use_lr_warmup):
      new_lr = trainOutputs.lr * PARAM.lr_halving_rate
      train_model.change_lr(sess, new_lr)

    # stop criterion
    if (epoch >= PARAM.max_epoch or
            model_abandon_time >= PARAM.max_model_abandon_time):
      misc_utils.print_log("\n\n", train_log_file, no_time=True)
      msg = "Training finished, final learning rate %e.\n" % trainOutputs.lr
      tf.logging.info(msg)
      misc_utils.print_log(msg, train_log_file)
      break

  sess.close()
  misc_utils.print_log("\n", train_log_file, no_time=True)
  msg = '################### Training Done. ###################\n'
  misc_utils.print_log(msg, train_log_file)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
