import os
import re
import shutil
import subprocess
import sys

import typing

tests_root = os.path.dirname(os.path.abspath(__file__))
distribution_root = os.path.dirname(tests_root)


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
    tests = os.sep.join([distribution_root, "tests_{}".format(hash_6)])

    # Copy tests

    shutil.copytree(tests_root, tests)
    tests = shutil.move(tests, tests_root)
    print("[ Test files copied to {}. ]".format(tests))

    # Get the path of virtualenvwrapper.sh

    status, output = subprocess.getstatusoutput(
        " ".join(["which", "virtualenvwrapper.sh"])
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

        ("cd", tests_root),

        ("source {} && workon {} && python"
         .format(vew_script, virtual_env),
         "-m", os.path.basename(tests)),

        ("source {} && rmvirtualenv".format(vew_script),
         virtual_env)
    ]

    for cmd in command_series:
        cmd_str = " ".join(cmd)

        status = execute_and_print(cmd_str)

        if status != 0:
            rm_env = " ".join(command_series[-1])
            status = execute_and_print(rm_env)
            if status == 0:
                print("[ {} executed successfully. ]".format(rm_env))
            shutil.rmtree(tests)
            print("[ Temporary test directory {} removed. ]".format(tests))
            return

    print("[ Temporary test directory {} removed. ]".format(tests))
    shutil.rmtree(tests)


if __name__ == "__main__":
    install_test(distribution_root, sys.executable)
