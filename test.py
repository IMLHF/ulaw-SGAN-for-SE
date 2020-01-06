import time
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def sub_processor(file_part, param1, param2):
  sleep_time = np.random.randint(3,6)
  ans = param1 + param2 + int(file_part.split('.')[0])
  time.sleep(sleep_time)
  with open(file_part, 'w') as fw:
    fw.write(str(ans))
  return ans

def multi_process():
  n_processor = 5
  files = ["0.txt","1.txt","2.txt","3.txt","4.txt","5.txt","6.txt","7.txt","8.txt","9.txt"]
  func = partial(sub_processor, param1=0, param2=1)
  job = Pool(n_processor).imap(func, files)
  ans_list = list(tqdm(job, "Testing", len(files), unit="test one file part", ncols=60))
  print(ans_list)

if __name__ == "__main__":
  multi_process()
