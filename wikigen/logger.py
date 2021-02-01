import os
import hashlib
import json
import subprocess
import socket
from datetime import datetime
from copy import copy
import shutil

# from .gsheets import GsheetsClient

try:
    import dataset as dt

    DATASET_LIB_AVAILABLE = True
except ImportError:
    print(
        "Warning: dataset package not available. Install it to save run "
        "info in sqlite database"
    )
    DATASET_LIB_AVAILABLE = False

from .settings import DATABASE_CONNECTION_STRING, PARAM_IGNORE_LIST


def get_server_name():
    try:
        server_name = socket.gethostname()
        return server_name
    except IOError as e:
        print("Server name not found. Creating run hash without it.")


def get_commit_hash():
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return output.decode()
    except subprocess.CalledProcessError:
        print("Current project is not a git repo, omitting commit hash.")
        return ""


def complete_params(args, model_name):

    new_args = copy(args)

    # For hash based instead of datetime based monitoring
    hashargs = {
        arg: val for arg, val in vars(args).items() if arg not in PARAM_IGNORE_LIST
    }

    hash = hashlib.sha1(
        json.dumps(hashargs, sort_keys=True).encode("utf8")
    ).hexdigest()[:8]

    new_args.hash = hash

    new_args.model_name = model_name
    new_args.host_name = get_server_name()
    new_args.commit = get_commit_hash()

    new_args.run_day = datetime.now().strftime("%Y_%m_%d")
    new_args.run_time = datetime.now().strftime("%H_%M_%S")
    new_args.run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_args.results_path = os.path.join(
        *(args.results_path, model_name, args.dataset, hash)
    )

    return vars(new_args)


class Logger(object):

    WRITE_MODES = ["FILE", "DATABASE", "BOTH", "NONE"]

    def __init__(
        self, args, model_name="", write_mode="BOTH", progress_bar=None, hash_value=None
    ):
        """Class in charge of saving info about a model being trained
           args: argparse.Namespace """
        if write_mode not in self.WRITE_MODES:
            raise ValueError(
                "write_mode not recognized, try using one of the"
                "following: {}".format(self.WRITE_MODES)
            )

        self.model_name = model_name
        self.write_mode = write_mode
        if self.write_mode == "NONE":
            self.write_mode = None

        if hash_value:
            self.hash = hash_value
            self.args = args
        else:
            self.args = complete_params(args, model_name)
            self.hash = self.args["hash"]

        self.run_savepath = self.args["results_path"]

        self._start_writing()

    def __getitem__(self, item):
        return self.args[item]

    def _start_writing(self):

        write_mode = self.write_mode
        if not DATASET_LIB_AVAILABLE:
            write_mode = "FILE"

        if write_mode == "FILE" or write_mode == "BOTH":

            if os.path.isdir(self.run_savepath):
                if not os.path.exists(
                    os.path.join(self.run_savepath, "model.pth.best")
                ):
                    print("Continuing previously failed run...")
                else:
                    if self.args["overwrite"]:
                        print("Existing model found, overwriting.")
                        shutil.rmtree(self.run_savepath, ignore_errors=True)
                        try:
                            os.makedirs(self.run_savepath)
                        except OSError:
                            pass
                    else:
                        print("Existing model found, add '--overwrite' to start over.")
                        exit()
            else:
                try:
                    os.makedirs(self.run_savepath)
                except OSError as e:
                    print(e)

            hyperparam_file = os.path.join(self.run_savepath, "hyperparams.json")
            with open(hyperparam_file, "w") as f:
                f.write(json.dumps(self.args, sort_keys=True))

        if write_mode == "DATABASE" or write_mode == "BOTH":
            db = dt.connect(DATABASE_CONNECTION_STRING)
            runs_table = db["runs"]
            runs_table.upsert(self.args, keys="hash")

        if write_mode is None:
            pass

        if write_mode not in ("FILE", "DATABASE", "BOTH", None):
            raise ValueError(
                "{} mode not recognized. Try with FILE, DATABASE "
                "or BOTH".format(write_mode)
            )

    # def write_architecture(self, model):
    #     architecture_filename = os.path.join(self.run_savepath, 'architecture.txt')
    #     with open(architecture_filename, 'w', encoding='utf8') as f:
    #         f.write(model)

    def _update_in_db(self, datadict):
        """expect a dictionary with the data to insert to the current run"""
        db = dt.connect(DATABASE_CONNECTION_STRING)
        runs_table = db["runs"]
        datadict["hash"] = self.hash
        runs_table.update(datadict, keys=["hash"])

    def update_results(self, datadict):
        """datadict: python dict"""
        if self.write_mode == "DATABASE" or self.write_mode == "BOTH":
            self._update_in_db(datadict)

        # TODO: implement option to write and update results to a file other
        # than a sqlite database

    def read_from_database(self):
        db = dt.connect(DATABASE_CONNECTION_STRING)
        runs_table = db["runs"]
        datadict = runs_table.find_one(hash=self.hash)
        return datadict

    # def insert_in_googlesheets(self):
    #     if self.write_mode == 'DATABASE' or self.write_mode == 'BOTH':
    #         print('Saving results in Google Spreadsheet')
    #         datadict = self.read_from_database()
    #         gsheets = GsheetsClient()
    #         gsheets.worksheet(config.SERVER_NAME).insert(datadict)

    # def write_current_run_details(self, model=None):
    #     if not os.path.exists(config.LOG_PATH):
    #         os.makedirs(config.LOG_PATH)
    #     run_details_filename = os.path.join(config.LOG_PATH, 'hyperparams.tmp')
    #     hyperparams = sorted(self.args.items())
    #     with open(run_details_filename, 'w', encoding='utf-8') as f:
    #         for k, v in hyperparams:
    #             # print('{}: {}'.format(k, v))
    #             f.write('{}: {}\n'.format(k, v))
    #
    #     if model:
    #         architecture_filename = os.path.join(config.LOG_PATH, 'architecture.tmp')
    #         with open(architecture_filename, 'w', encoding='utf8') as f:
    #             f.write(model)

    # def torch_save_file(self, filename, obj, path_override=None,
    #                     progress_bar=None):
    #     """progress bar should be a tqdm instance with the `write` method"""
    #     if not os.path.isdir(self.run_savepath):
    #         os.makedirs(self.run_savepath)
    #     savepath = os.path.join(self.run_savepath, filename)
    #     savepath = path_override if path_override else savepath
    #     torch.save(obj, savepath)
    #     if progress_bar:
    #         progress_bar.write(f'File saved in {savepath}')
    #     return
    #
