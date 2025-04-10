import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 顶层函数，multiprocessing 才能 pickle
def _worker(args):
    fn, d, kwargs = args
    return fn(*d, **kwargs)  # d 必须是 tuple；如果是单参数就传成 (d,) 即可


from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    if n_jobs is None:
        n_jobs = cpu_count()

    # 定义一个真正可以 pickling 的函数，避免 lambda 引起问题
    def _wrapped(d):
        return pickleable_fn(*d, **kwargs)

    # tqdm 外部包裹，不要嵌入 generator 里
    data_iter = list(tqdm(data, desc=desc))

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(_wrapped)(d) for d in data_iter
    )
    return results

    
# def pmap_multi(pickleable_fn, data, n_jobs=None, desc=None, **kwargs):
#     """
#     Parallel map using multiprocessing.Pool, supports multi-argument unpacking.

#     Parameters
#     ----------
#     pickleable_fn : callable
#         Function to apply.
#     data : list of tuple
#         Inputs to apply the function on. Each element is a tuple of positional arguments.
#     n_jobs : int
#         Number of processes. Defaults to all available CPUs.
#     desc : str
#         tqdm description.
#     kwargs : dict
#         Extra keyword arguments passed to the function.

#     Returns
#     -------
#     List of function outputs.
#     """
#     if n_jobs is None:
#         n_jobs = cpu_count()

#     # 构建 [(fn, d_i, kwargs)] 列表
#     task_args = [(pickleable_fn, d if isinstance(d, tuple) else (d,), kwargs) for d in data]

#     with Pool(processes=n_jobs) as pool:
#         results = list(tqdm(pool.imap(_worker, task_args), total=len(data), desc=desc))

#     return results


def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval


class RectifiedFlow():
  def __init__(self, model=None, num_steps=1000):
    self.model = model
    self.N = num_steps

  def get_train_tuple(self, z0=None, z1=None, t=None, batch_id=None):
    dtype = z0.dtype
    if batch_id is None:
        t = torch.rand((z1.shape[0], 1, 1), device=z0.device, dtype=dtype)
    else:
        t = torch.rand((batch_id.unique().shape[0], 1, 1), device=z0.device, dtype=dtype)
        t = t[batch_id]
    z_t =  t * z1 + (1.-t) * z0
    # target = z1 - z0
    target = z1 - z_t

    return z_t, t, target

  @torch.no_grad()
  def sample_ode(self, z0=None,  N=None, batch_id=None, chain_encoding=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    norm_max = torch.ones_like(batch_id, device=z0.device)[:,None] * 10.

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1,1), device=z0.device) * i / N
      z1_hat, all_preds, vq_los = self.model(chain_encoding, batch_id, z, t, norm_max)
      z = z.detach().clone() + (z1_hat/norm_max[...,None]-z)/(1-t) * dt
    #   z =  t * z1_hat + (1.-t) * z0

      traj.append(z.detach().clone()*norm_max[...,None])

    return traj


def flatten_dict(d, parent_key='', sep='.', level=0):
    """
    递归地将嵌套字典拉平为一个单层字典，取消第一级父键。

    :param d: 输入的嵌套字典
    :param parent_key: 父键（用于递归）
    :param sep: 键之间的分隔符，默认为点号 '.'
    :param level: 当前递归的层级（用于取消第一级父键）
    :return: 拉平后的单层字典
    """
    items = {}
    for k, v in d.items():
        # 构建新的键
        if level <=1:
            new_key = k  # 第一级取消父键
        else:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # 如果值是字典，递归拉平
            items.update(flatten_dict(v, new_key, sep=sep, level=level + 1))
        else:
            # 否则直接添加到结果中
            items[new_key] = v
    return items
