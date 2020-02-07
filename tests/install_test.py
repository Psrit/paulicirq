import os
import re
import shutil
import subprocess
import sys

import typing

tests_root = os.path.dirname(os.path.abspath(__file__))
distribution_root = os.path.dirname(tests_root)


class VirtualEnvWrapperNotFoundError(Exception):
    def __init__(self, msg):
        self.msg = msg


def random_sha1_by_time(length=9):
    import hashlib
    import datetime
    return hashlib.sha1(
        str(datetime.datetime.now().timestamp()).encode()
    ).hexdigest()[:length]


def execute_and_print(
        cmd_str,
        stdout: typing.TextIO = sys.stdout,
        stderr: typing.TextIO = sys.stderr
) -> int:
    print("[ {} ]".format(cmd_str))
    with subprocess.Popen(cmd_str,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          bufsize=1, shell=True) as popen:

        # Redirect stdout
        while popen.poll() is None:  # None: running
            r = popen.stdout.readline().decode(sys.getdefaultencoding())
            stdout.write(r)

        # Redirect stderr
        status = popen.poll()
        if status != 0:  # if there is an error
            err = popen.stderr.read().decode(sys.getdefaultencoding())
            stderr.write(err)

        return status


def install_test(package_path, python):
    # Get Python version

    status, output = subprocess.getstatusoutput(
        " ".join([sys.executable, "--version"])
    )
    python_version = re.search(
        re.compile(r"(?P<version>[\d.]+)"), output
    ).group("version")

    # Generate random name of virtualenv and tests

    hash_6 = random_sha1_by_time(6)
    virtual_env = "pkgtest-paulicirq-{}-{}".format(
        python_version,
        hash_6
    )
    tests_dir = os.sep.join([distribution_root, "tests_{}".format(hash_6)])
    os.makedirs(tests_dir, exist_ok=False)
    tests = os.sep.join([tests_dir, "tests"])

    # Copy tests

    shutil.copytree(tests_root, tests)
    print("[ Test files copied to \"{}\". ]".format(tests))

    # Change current working directory

    _cwd = os.getcwd()
    os.chdir(tests_dir)
    print("[ Change current working directory to: {} ]"
          .format(os.getcwd()))

    # Get the path of virtualenvwrapper.sh

    output = shutil.which("virtualenvwrapper.sh")
    if output is None:
        raise VirtualEnvWrapperNotFoundError(
            msg="virtualenvwrapper is required for the installation test."
        )

    vew_script = output

    # Command series

    command_series = [
        ("source {} && mkvirtualenv".format(vew_script),
         virtual_env, "--python", python),

        # ("pwd",),

        ("source {} && workon {} && python"
         .format(vew_script, virtual_env),
         "-m", "pip", "install", "--default-timeout=100",
         "-i https://pypi.tuna.tsinghua.edu.cn/simple", package_path),

        ("source {} && workon {} && python"
         .format(vew_script, virtual_env),
         "-m", "pip", "list"),

        ("source {} && workon {} && python"
         .format(vew_script, virtual_env),
         "-m", os.path.basename(tests)),

        ("source {} && rmvirtualenv".format(vew_script),
         virtual_env)
    ]

    def _clean():
        os.chdir(_cwd)
        shutil.rmtree(tests_dir)
        print("[ Temporary test directory {} removed. ]".format(tests_dir))

    for cmd in command_series:
        cmd_str = " ".join(cmd)

        status = execute_and_print(cmd_str)

        if status != 0:
            rm_env = " ".join(command_series[-1])
            status = execute_and_print(rm_env)
            if status == 0:
                print("[ {} executed successfully. ]".format(rm_env))

            _clean()
            return

    _clean()


if __name__ == "__main__":
    install_test(distribution_root, sys.executable)
