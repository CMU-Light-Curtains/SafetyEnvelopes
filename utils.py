from collections import deque
import datetime
import functools
from typing import Union, List, Any, NoReturn

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import sacred
from lc_planner.planner import PlannerRT
from termcolor import colored
import time

########################################################################################################################
# region: Sacred
########################################################################################################################


INGREDIENTS = []


class Ingredient(sacred.Ingredient):
    def __init__(self, path):
        super().__init__(path, ingredients=INGREDIENTS)
        INGREDIENTS.append(self)


class Experiment(sacred.Experiment):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, ingredients=INGREDIENTS, **kwargs)
        INGREDIENTS.append(self)

# endregion
########################################################################################################################
# region: Timer
########################################################################################################################


"""
Usage:
    from utils import timer

    # this may be in a loop
    with timer.time_as("loading images"):
        /* code that loads images */
    
    # this may be in a loop
    with timer.time_as("computation"):
        /* code that does computation */
    
    timer.print_stats()
"""


class Timer:
    def __init__(self):
        self.ttime = {}  # total time for every key
        self.ttimesq = {}  # total time-squared for every key
        self.titer = {}  # total number of iterations for every key

    def _add(self, key, time_):
        if key not in self.ttime:
            self.ttime[key] = 0
            self.ttimesq[key] = 0
            self.titer[key] = 0

        self.ttime[key] += time_
        self.ttimesq[key] += time_ * time_
        self.titer[key] += 1

    def print_stats(self):
        print("TIMER STATS:")
        word_len = max([len(k) for k in self.ttime.keys()]) + 8
        for key in self.ttime:
            ttime_, ttimesq_, titer_ = self.ttime[key], self.ttimesq[key], self.titer[key]
            mean = ttime_ / titer_
            std = np.sqrt(ttimesq_ / titer_ - mean * mean)
            interval = 1.96 * std
            print(f"{key.rjust(word_len)}: {mean:.3f}s ± {interval:.3f}s")

    class TimerContext:
        def __init__(self, timer, key):
            self.timer = timer
            self.key = key

            self._stime = 0
            self._etime = 0

        def __enter__(self):
            self._stime = time.time()

        def __exit__(self, type, value, traceback):
            self._etime = time.time()
            time_ = self._etime - self._stime
            self.timer._add(self.key, time_)

    def time_as(self, key):
        return Timer.TimerContext(self, key)

    def time_fn(self, name):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                with self.time_as(name):
                    ret = function(*args, **kwargs)
                ttime_   = self.ttime[name]
                titer_   = self.titer[name]
                avg_time = ttime_ / titer_
                print(f"Avg. {name} time is {datetime.timedelta(seconds=round(avg_time))}s")
                return ret
            return wrapper
        return decorator


timer = Timer()

# endregion
########################################################################################################################
# region: Meters
########################################################################################################################


# copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '[ {name}: {avg' + self.fmt + '} ]'
        return fmtstr.format(**self.__dict__)

# endregion
########################################################################################################################
# region: History class
########################################################################################################################


class History:
    """
    This class maintains a finite history of past observations in a FIFO manner.
    When there are fewer observations than the history size, it repeats past observations.
    """
    def __init__(self,
                 size: int):
        assert size > 0
        self.size = size
        self.queue = deque([None] * self.size, maxlen=self.size)

    def add(self, obs: Any) -> NoReturn:
        self.queue.append(obs)  # if deque is full, it will remove elements from the left

        # this is meant for the first add() call
        # the while loop will fill the queue with the same element and replace all None's
        while self.queue[0] is None:
            self.queue.append(obs)

    def get(self) -> List[Any]:
        """Returns a list of length self.size, even if queue is not full"""
        assert self.queue[0] is not None, "Must call History.add() at least once before calling History.get()"
        return list(self.queue)

    def clear(self) -> NoReturn:
        # fill all elements of the queue with None's
        for i in range(self.size):
            self.queue[i] = None

    def __str__(self):
        return str(self.queue)

    def __repr__(self):
        return super().__repr__() + '\n' + str(self)

# endregion
########################################################################################################################
# region: Light curtain utils
########################################################################################################################


def valid_curtain_behind_frontier(planner: PlannerRT,
                                  frontier: np.ndarray,
                                  debug: bool = False):
    """Computes a valid curtain that lies strictly behind the current frontier using the planner

    Args:
        planner: (lc_planner.planner.PlannerRT) PlannerRT initialized with ranges that performs minimization.
        frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correspond to a
                    valid curtain.
        debug (bool): whether to debug or not.
    Returns:
        curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that corresponds to a valid curtain.
    """
    # construct cmap
    ranges = planner.ranges  # (R,)
    safe_mask = ranges.reshape(-1, 1) <= frontier  # (R, C)
    distances = np.abs(ranges.reshape(-1, 1) - frontier)  # (R, C)

    # cost map = negative distance in safety region and negative infinity outside
    cmap = -np.inf * np.ones_like(distances)  # (R, C)
    safe_mask_i, safe_mask_j = np.where(safe_mask)  # both are (N,)
    cmap[safe_mask_i, safe_mask_j] = -distances[safe_mask_i, safe_mask_j]  # (R, C)

    design_pts = planner.get_design_points(cmap)  # (C, 2)
    assert design_pts.shape == (planner.num_camera_angles, 2)

    if debug:
        unsafe_mask_i, unsafe_mask_j = np.where(np.logical_not(safe_mask))
        cmap[unsafe_mask_i, unsafe_mask_j] = 0
        cmap[safe_mask_i, safe_mask_j] = 100 + cmap[safe_mask_i, safe_mask_j]
        planner._visualize_curtain_rt(cmap, design_pts, show=False)
        new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
        thetas = np.arctan2(new_z, new_x)
        old_x, old_z = frontier * np.cos(thetas), frontier * np.sin(thetas)
        plt.plot(old_x, old_z, c='r', linewidth=0.5)
        plt.ylim(0, 30)
        plt.xlim(-10, 10)
        plt.show()

    # compute curtain from design points
    curtain = np.linalg.norm(design_pts, axis=1)  # (C,)

    # assert that curtain lies completely behind planner
    if not np.all(curtain <= frontier + 1e-3):
        # debug this
        raise AssertionError("planner doesn't place curtain completely behind frontier.")

    return curtain


def valid_curtain_close_to_frontier(planner: PlannerRT,
                                    frontier: np.ndarray,
                                    debug: bool = False):
    """Computes a valid curtain that lies as close to the curtain (L1 distance) as possible using the planner
    Args:
        planner: (lc_planner.planner.PlannerRT) PlannerRT initialized with ranges that performs minimization.
        frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correspond to a
                    valid curtain.
        debug (bool): whether to debug or not.
    Returns:
        curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that corresponds to a valid curtain.
    """
    # cost map = negative L1 distance between frontier and range location
    ranges = planner.ranges  # (R,)
    cmap = -np.abs(ranges.reshape(-1, 1) - frontier)  # (R, C)

    design_pts = planner.get_design_points(cmap)  # (C, 2)
    assert design_pts.shape == (planner.num_camera_angles, 2)

    if debug:
        planner._visualize_curtain_rt(cmap, design_pts, show=False)
        new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
        thetas = np.arctan2(new_z, new_x)
        old_x, old_z = frontier * np.cos(thetas), frontier * np.sin(thetas)
        plt.plot(old_x, old_z, c='r', linewidth=0.5)
        plt.ylim(0, 30)
        plt.xlim(-10, 10)
        plt.show()

    # compute curtain from design points
    curtain = np.linalg.norm(design_pts, axis=1)  # (C,)

    return curtain


def design_pts_from_ranges(ranges: np.ndarray,
                           thetas: np.ndarray):
    """
        Args:
            ranges (np.ndarray, shape=(C,), dtype=np.float32): range per camera ray
            thetas (np.ndarray, shape=(C,), dtype=np.float32): in degrees and in increasing order in [-fov/2, fov/2]
        Returns:
            design_pts: (np.ndarray, shape=(C, 2), dtype=np.float32) design points corresponding to frontier.
                        - Axis 1 channels denote (x, z) in camera frame.
        """
    x = ranges * np.sin(np.deg2rad(thetas))
    z = ranges * np.cos(np.deg2rad(thetas))
    design_pts = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
    return design_pts

# endregion
########################################################################################################################
# region: Sacred utils
########################################################################################################################


def get_sacred_artifact_from_mongodb(run_id, name):
    """
    Get artifact from MongoDB

    Args:
        run_id (int): id of the run
        name (string): name of the artifact saved

    Returns:
        file: a file-like object with a read() function
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    fs = gridfs.GridFS(client.sacred)

    runs = client.sacred.runs
    run_entry = runs.find_one({'_id': run_id})
    artifacts = run_entry["artifacts"]
    artifacts = list(filter(lambda entry: entry["name"] == "actor_weights", artifacts))
    assert len(artifacts) == 1,\
        str(f"Number of artifacts with run_id={run_id} and name={name} is {len(artifacts)} instead of 1")
    file_id = artifacts[0]['file_id']
    file = fs.get(file_id)  # this is a file-like object that has a read() method

    return file

# endregion
########################################################################################################################
# Display utils
########################################################################################################################


def cprint(s: str, color: str = "yellow"):
    print(colored(s, color))


def pprint(s: Union[str, List[str]],
           color: str = "yellow"):
    """
    If the list of strings is [s1, s2, s3, s4], the output will look like:

    ========
    s1
    ========
    s2
    s3
    s4
    ========

    The number of equal to sign is the length of the longest string
    """
    if type(s) != list:
        s = [s]
    max_len = max([len(e) for e in s]) + 1
    horizontal_line = "=" * max_len

    cprint(horizontal_line, color)
    cprint(s[0], color)
    cprint(horizontal_line, color)
    if len(s) > 1:
        cprint('\n'.join(s[1:]), color)
        cprint(horizontal_line, color)

# endregion
########################################################################################################################
# Registering classes
########################################################################################################################


REGISTERED_CLASSES = {}


def register_class(group_name):
    def wrapper(cls):
        global REGISTERED_CLASSES
        cls_name = cls.__name__
        if group_name not in REGISTERED_CLASSES:
            REGISTERED_CLASSES[group_name] = {}
        group_dict = REGISTERED_CLASSES[group_name]
        assert cls_name not in group_dict, f"{cls_name} already exists for group {group_name}"
        group_dict[cls_name] = cls
        return cls
    return wrapper


def get_class(group_name, cls_name):
    global REGISTERED_CLASSES
    assert group_name in REGISTERED_CLASSES, f"group \"{group_name}\" not available"
    group_dict = REGISTERED_CLASSES[group_name]
    assert cls_name in group_dict, f"available classes in group {group_name}: {group_dict}"
    return group_dict[cls_name]

########################################################################################################################
