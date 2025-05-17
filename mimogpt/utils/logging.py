import logging
import sys
import os
from datetime import datetime


logger_name = "MGM"


class LOGGER(logging.Logger):
    def __init__(self, logger_name, rank=0):
        super(LOGGER, self).__init__(logger_name)
        self.log_fn = None
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s]:%(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = "rank_{}.log".format(rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.log_fn = log_fn

    def copy_log_to_s3(self, train_url):
        try:
            import moxing as mox

            roma_log_fp = os.path.join(train_url, self.log_fn)
            roma_log_dirname = os.path.dirname(roma_log_fp)
            if not mox.file.exists(roma_log_dirname):
                mox.file.make_dirs(roma_log_dirname)
            mox.file.copy(self.log_fn, roma_log_fp)
        except:
            pass

    def info(self, msg="", *args, **kwargs):  # set msg default value, since usage: print(file=file)
        if self.isEnabledFor(logging.INFO):
            if len(args) > 0:
                if "%" in msg:
                    msg = str(msg) % args
                else:
                    msg = " ".join([str(msg)] + [str(x) for x in args])

            if "file" in kwargs.keys():
                kwargs.pop("file")

            if "end" in kwargs.keys():
                msg += kwargs["end"]
                kwargs.pop("end")

            if "flush" in kwargs.keys():
                kwargs.pop("flush")

            self._log(logging.INFO, msg, (), **kwargs)

    def save_args(self, args):
        self.info("Args:")
        if isinstance(args, (list, tuple)):
            for value in args:
                self.info("--> {}".format(value))
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                self.info("--> {}: {}".format(key, args_dict[key]))
        self.info("")


def get_logger(path, rank=0):
    logger = LOGGER(logger_name, rank)
    logger.setup_logging_file(path, rank)
    return logger
