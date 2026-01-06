"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 100
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        """
        输入:
            kvs: 键值对字典。
        输出:
            无
        作用: 写入键值对。
        逻辑:
        抽象方法。
        """
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        """
        输入:
            seq: 序列。
        输出:
            无
        作用: 写入序列。
        逻辑:
        抽象方法。
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        """
        输入:
            filename_or_file: 文件名或文件对象。
        输出:
            无
        作用: 初始化人类可读输出格式。
        逻辑:
        如果是字符串，打开文件；否则直接使用文件对象。
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        """
        输入:
            kvs: 键值对字典。
        输出:
            无
        作用: 写入键值对。
        逻辑:
        1. 格式化键值对为字符串。
        2. 计算最大宽度。
        3. 写入表格形式的数据。
        4. 刷新文件。
        """
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        """
        输入:
            s: 字符串。
        输出:
            截断后的字符串。
        作用: 截断过长的字符串。
        逻辑:
        如果长度超过 30，截断并添加 "..."。
        """
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        """
        输入:
            seq: 序列。
        输出:
            无
        作用: 写入序列。
        逻辑:
        遍历序列并写入，元素之间用空格分隔。
        """
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        输入:
            无
        输出:
            无
        作用: 关闭文件。
        逻辑:
        如果是自己打开的文件，则关闭。
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        输入:
            filename: 文件名。
        输出:
            无
        作用: 初始化 JSON 输出格式。
        逻辑:
        打开文件。
        """
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        """
        输入:
            kvs: 键值对字典。
        输出:
            无
        作用: 写入键值对。
        逻辑:
        1. 处理 numpy 类型。
        2. 写入 JSON 字符串。
        3. 刷新文件。
        """
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        """
        输入:
            无
        输出:
            无
        作用: 关闭文件。
        逻辑:
        关闭文件。
        """
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        输入:
            filename: 文件名。
        输出:
            无
        作用: 初始化 CSV 输出格式。
        逻辑:
        打开文件，初始化键列表。
        """
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        """
        输入:
            kvs: 键值对字典。
        输出:
            无
        作用: 写入键值对。
        逻辑:
        1. 检查是否有新键。
        2. 如果有新键，重写表头和旧数据。
        3. 写入新数据。
        """
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        """
        输入:
            dir: 目录路径。
        输出:
            无
        作用: 初始化 TensorBoard 输出格式。
        逻辑:
        创建目录，初始化 TensorBoard writer。
        """
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        """
        输入:
            kvs: 键值对字典。
        输出:
            无
        作用: 写入键值对到 TensorBoard。
        逻辑:
        创建 Summary，写入 Event。
        """
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        """
        输入:
            无
        输出:
            无
        作用: 关闭 writer。
        逻辑:
        关闭 writer。
        """
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    """
    输入:
        format: 格式字符串。
        ev_dir: 事件目录。
        log_suffix: 日志后缀。
    输出:
        KVWriter: 输出格式对象。
    作用: 创建输出格式对象。
    逻辑:
    根据格式字符串创建相应的输出格式对象。
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    输入:
        key: 键。
        val: 值。
    输出:
        无
    作用: 记录键值对。
    逻辑:
    调用当前 logger 的 logkv 方法。
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    输入:
        key: 键。
        val: 值。
    输出:
        无
    作用: 记录键值对的平均值。
    逻辑:
    调用当前 logger 的 logkv_mean 方法。
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    输入:
        d: 键值对字典。
    输出:
        无
    作用: 记录字典中的键值对。
    逻辑:
    遍历字典，调用 logkv。
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    输入:
        无
    输出:
        dict: 键值对字典。
    作用: 转储当前迭代的所有诊断信息。
    逻辑:
    调用当前 logger 的 dumpkvs 方法。
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    """
    输入:
        无
    输出:
        dict: 键值对字典。
    作用: 获取当前迭代的所有诊断信息。
    逻辑:
    返回当前 logger 的 name2val。
    """
    return get_current().name2val


def log(*args, level=INFO):
    """
    输入:
        *args: 日志内容。
        level: 日志级别。
    输出:
        无
    作用: 记录日志。
    逻辑:
    调用当前 logger 的 log 方法。
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    """
    输入:
        *args: 日志内容。
    输出:
        无
    作用: 记录调试日志。
    逻辑:
    调用 log 方法，级别为 DEBUG。
    """
    log(*args, level=DEBUG)


def info(*args):
    """
    输入:
        *args: 日志内容。
    输出:
        无
    作用: 记录信息日志。
    逻辑:
    调用 log 方法，级别为 INFO。
    """
    log(*args, level=INFO)


def warn(*args):
    """
    输入:
        *args: 日志内容。
    输出:
        无
    作用: 记录警告日志。
    逻辑:
    调用 log 方法，级别为 WARN。
    """
    log(*args, level=WARN)


def error(*args):
    """
    输入:
        *args: 日志内容。
    输出:
        无
    作用: 记录错误日志。
    逻辑:
    调用 log 方法，级别为 ERROR。
    """
    log(*args, level=ERROR)


def set_level(level):
    """
    输入:
        level: 日志级别。
    输出:
        无
    作用: 设置日志级别。
    逻辑:
    调用当前 logger 的 set_level 方法。
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    """
    输入:
        comm: MPI 通信器。
    输出:
        无
    作用: 设置 MPI 通信器。
    逻辑:
    调用当前 logger 的 set_comm 方法。
    """
    get_current().set_comm(comm)


def get_dir():
    """
    输入:
        无
    输出:
        str: 日志目录。
    作用: 获取日志目录。
    逻辑:
    调用当前 logger 的 get_dir 方法。
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    """
    输入:
        scopename: 作用域名称。
    输出:
        无
    作用: 性能分析上下文管理器。
    逻辑:
    记录代码块执行时间。
    """
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    输入:
        n: 名称。
    输出:
        function: 装饰器。
    作用: 性能分析装饰器。
    逻辑:
    使用 profile_kv 记录函数执行时间。
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    """
    输入:
        无
    输出:
        Logger: 当前 logger 对象。
    作用: 获取当前 logger。
    逻辑:
    如果当前 logger 为空，配置默认 logger。
    """
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        """
        输入:
            dir: 日志目录。
            output_formats: 输出格式列表。
            comm: MPI 通信器。
        输出:
            无
        作用: 初始化 Logger。
        逻辑:
        初始化变量。
        """
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        """
        输入:
            key: 键。
            val: 值。
        输出:
            无
        作用: 记录键值对。
        逻辑:
        更新 name2val。
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        输入:
            key: 键。
            val: 值。
        输出:
            无
        作用: 记录键值对的平均值。
        逻辑:
        更新 name2val 和 name2cnt，计算平均值。
        """
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        """
        输入:
            无
        输出:
            dict: 键值对字典。
        作用: 转储所有诊断信息。
        逻辑:
        1. 如果有 MPI，计算加权平均。
        2. 写入所有输出格式。
        3. 清空 name2val 和 name2cnt。
        """
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        """
        输入:
            *args: 日志内容。
            level: 日志级别。
        输出:
            无
        作用: 记录日志。
        逻辑:
        如果级别满足要求，调用 _do_log。
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        """
        输入:
            level: 日志级别。
        输出:
            无
        作用: 设置日志级别。
        逻辑:
        更新 level。
        """
        self.level = level

    def set_comm(self, comm):
        """
        输入:
            comm: MPI 通信器。
        输出:
            无
        作用: 设置 MPI 通信器。
        逻辑:
        更新 comm。
        """
        self.comm = comm

    def get_dir(self):
        """
        输入:
            无
        输出:
            str: 日志目录。
        作用: 获取日志目录。
        逻辑:
        返回 dir。
        """
        return self.dir

    def close(self):
        """
        输入:
            无
        输出:
            无
        作用: 关闭所有输出格式。
        逻辑:
        遍历 output_formats，调用 close。
        """
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        """
        输入:
            args: 日志内容。
        输出:
            无
        作用: 执行日志记录。
        逻辑:
        遍历 output_formats，如果是 SeqWriter，调用 writeseq。
        """
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import():
    """
    输入:
        无
    输出:
        int: 进程排名。
    作用: 获取 MPI 排名，不导入 mpi4py。
    逻辑:
    检查环境变量 PMI_RANK 或 OMPI_COMM_WORLD_RANK。
    check environment variables here instead of importing mpi4py
    to avoid calling MPI_Init() when this module is imported
    """
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
    输入:
        comm: MPI 通信器。
        local_name2valcount: 本地键值对计数。
    输出:
        dict: 加权平均后的键值对。
    作用: 计算 MPI 加权平均。
    逻辑:
    1. 收集所有节点的数据。
    2. 在 rank 0 计算加权平均。
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(dir='./results', format_strs=None, comm=None, log_suffix=""):
    """
    输入:
        dir: 日志目录。
        format_strs: 格式字符串列表。
        comm: MPI 通信器。
        log_suffix: 日志后缀。
    输出:
        无
    作用: 配置 logger。
    逻辑:
    1. 确定日志目录。
    2. 确定格式字符串。
    3. 创建输出格式对象。
    4. 创建并设置当前 Logger。
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            "./results",
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    """
    输入:
        无
    输出:
        无
    作用: 配置默认 logger。
    逻辑:
    调用 configure，并设置 Logger.DEFAULT。
    """
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    """
    输入:
        无
    输出:
        无
    作用: 重置 logger。
    逻辑:
    关闭当前 logger，恢复默认 logger。
    """
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    """
    输入:
        dir: 日志目录。
        format_strs: 格式字符串列表。
        comm: MPI 通信器。
    输出:
        无
    作用: 作用域配置 logger。
    逻辑:
    保存当前 logger，配置新 logger，退出时恢复。
    """
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

