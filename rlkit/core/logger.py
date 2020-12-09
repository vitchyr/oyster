"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib
import json
import pickle
import base64
import errno
import torch

from rlkit.core.tabulate import tabulate


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='w')


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def get_snapshot_gap():
    return _snapshot_gap


def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    if not _log_tabular_only:
        # Also log to stdout
        print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def save_extra_data(data, path='extra_data', ext='.pkl'):
    """
    Data saved here will always override the last entry

    :param data: Something pickle'able.
    """
    file_name = osp.join(_snapshot_dir, path + ext)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_table_dict():
    return dict(_tabular)


def get_table_key_set():
    return set(key for key, value in _tabular)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split('\n'):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd,
                                    fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)

def save_weights(weights, names):
    ''' save network weights to given paths '''
    # NOTE: breaking abstraction by adding torch dependence here
    for w, n in zip(weights, names):
        torch.save(w, n)

def save_itr_params(itr, params_dict):
    ''' snapshot model parameters '''
    # NOTE: assumes dict is ordered, should fix someday
    names = params_dict.keys()
    params = params_dict.values()
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            # save for every training iteration
            file_names = [osp.join(_snapshot_dir, n + '_itr_%d.pth' % itr) for n in names]
            save_weights(params, file_names)
        elif _snapshot_mode == 'last':
            # override previous params
            file_names = [osp.join(_snapshot_dir, n + '.pth') for n in names]
            save_weights(params, file_names)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_names = [osp.join(_snapshot_dir, n + '_itr_%d.pth' % itr) for n in names]
                save_weights(params, file_names)
        elif _snapshot_mode == "gap_and_last":
            if itr % _snapshot_gap == 0:
                file_names = [osp.join(_snapshot_dir, n + '_itr_%d.pth' % itr) for n in names]
                save_weights(params, file_names)
            file_names = [osp.join(_snapshot_dir, n + '.pth') for n in names]
            save_weights(params, file_names)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name}
        return json.JSONEncoder.default(self, o)


def log_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)


def record_tabular_misc_stat(key, values, placement='back'):
    if placement == 'front':
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
    if len(values) > 0:
        record_tabular(prefix + "Average" + suffix, np.average(values))
        record_tabular(prefix + "Std" + suffix, np.std(values))
        record_tabular(prefix + "Median" + suffix, np.median(values))
        record_tabular(prefix + "Min" + suffix, np.min(values))
        record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        record_tabular(prefix + "Average" + suffix, np.nan)
        record_tabular(prefix + "Std" + suffix, np.nan)
        record_tabular(prefix + "Median" + suffix, np.nan)
        record_tabular(prefix + "Min" + suffix, np.nan)
        record_tabular(prefix + "Max" + suffix, np.nan)


def setup_logger(
        logger,
        exp_name,
        base_log_dir,
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        tensorboard=False,
        unique_id=None,
        git_infos=None,
        script_name=None,
        run_id=None,
        push_prefix=True,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to
        based_log_dir/exp_name/exp_name.
    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.
    :param exp_name: The sub-directory for this specific experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :return:
    """
    logger.reset()
    variant = variant or {}
    unique_id = unique_id or str(uuid.uuid4())

    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(
            exp_name=exp_name,
            base_log_dir=base_log_dir,
            variant=variant,
            run_id=run_id,
            **create_log_dir_kwargs
        )

    if tensorboard:
        tensorboard_log_path = osp.join(log_dir, "tensorboard")
        logger.add_tensorboard_output(tensorboard_log_path)

    logger.log("Variant:")
    variant_to_save = variant.copy()
    variant_to_save['unique_id'] = unique_id
    variant_to_save['exp_name'] = exp_name
    variant_to_save['trial_name'] = log_dir.split('/')[-1]
    logger.log(
        json.dumps(ppp.dict_to_safe_json(variant_to_save, sort=True), indent=2)
    )
    variant_log_path = osp.join(log_dir, variant_log_file)
    logger.log_variant(variant_log_path, variant_to_save)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    if push_prefix:
        logger.push_prefix("[%s] " % exp_name)

    if git_infos:
        save_git_infos(git_infos, log_dir)
    if script_name:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


def save_git_infos(git_infos, log_dir):
    for (
            directory, code_diff, code_diff_staged, commit_hash, branch_name
    ) in git_infos:
        if directory[-1] == '/':
            diff_file_name = directory[1:-1].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                    directory[1:-1].replace("/", "-") + "_staged.patch"
            )
        else:
            diff_file_name = directory[1:].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                    directory[1:].replace("/", "-") + "_staged.patch"
            )
        if code_diff is not None and len(code_diff) > 0:
            with open(osp.join(log_dir, diff_file_name), "w") as f:
                f.write(code_diff + '\n')
        if code_diff_staged is not None and len(code_diff_staged) > 0:
            with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                f.write(code_diff_staged + '\n')
        with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
            f.write("directory: {}".format(directory))
            f.write('\n')
            f.write("git hash: {}".format(commit_hash))
            f.write('\n')
            f.write("git branch name: {}".format(branch_name))
            f.write('\n\n')


def create_log_dir(
        exp_name,
        base_log_dir,
        exp_id=0,
        seed=0,
        variant=None,
        trial_dir_suffix=None,
        add_time_suffix=True,
        include_exp_name_sub_dir=True,
        run_id=None,
):
    """
    Creates and returns a unique log directory.
    :param exp_name: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    if run_id is not None:
        exp_id = variant["exp_id"]
        if variant.get("num_exps_per_instance", 0) > 1:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
            trial_name = "run%s/id%s/%s--s%d" % (run_id, exp_id, timestamp, seed)
        else:
            trial_name = "run{}/id{}".format(run_id, exp_id)
    else:
        trial_name = create_trial_name(exp_name, exp_id=exp_id, seed=seed, add_time_suffix=add_time_suffix)
    if trial_dir_suffix is not None:
        trial_name = "{}-{}".format(trial_name, trial_dir_suffix)
    if include_exp_name_sub_dir:
        log_dir = osp.join(base_log_dir, exp_name.replace("_", "-"), trial_name)
    else:
        log_dir = osp.join(base_log_dir, trial_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_trial_name(exp_name, exp_id=0, seed=0, add_time_suffix=True):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_name:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if add_time_suffix:
        return "%s_%s_id%03d--s%d" % (exp_name, timestamp, exp_id, seed)
    else:
        return "%s_id%03d--s%d" % (exp_name, exp_id, seed)


logger = Logger()
