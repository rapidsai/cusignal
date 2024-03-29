# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys
import tempfile

EXPECTED_VERSION = "8.0.1"
VERSION_REGEX = re.compile(r"clang-format version ([0-9.]+)")
# NOTE: populate this list with more top-level dirs as we add more of them to
# the cudf repo
DEFAULT_DIRS = [
    "cpp/benchmarks",
    "cpp/include",
    "cpp/include/cudf",
    "cpp/include/nvtext",
    "cpp/src",
    "cpp/tests",
]


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-format on a project")
    argparser.add_argument(
        "-dstdir",
        type=str,
        default=None,
        help="Directory to store the temporary outputs of"
        " clang-format. If nothing is passed for this, then"
        " a temporary dir will be created using `mkdtemp`",
    )
    argparser.add_argument(
        "-exe",
        type=str,
        default="clang-format",
        help="Path to clang-format exe",
    )
    argparser.add_argument(
        "-inplace",
        default=False,
        action="store_true",
        help="Replace the source files itself.",
    )
    argparser.add_argument(
        "-regex",
        type=str,
        default=r"[.](cu|cuh|h|hpp|cpp|inl)$",
        help="Regex string to filter in sources",
    )
    argparser.add_argument(
        "-ignore",
        type=str,
        default=r"cannylab/bh[.]cu$",
        help="Regex used to ignore files from matched list",
    )
    argparser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Print verbose messages",
    )
    argparser.add_argument(
        "dirs", type=str, nargs="*", help="List of dirs where to find sources"
    )
    args = argparser.parse_args()
    args.regex_compiled = re.compile(args.regex)
    args.ignore_compiled = re.compile(args.ignore)
    if args.dstdir is None:
        args.dstdir = tempfile.mkdtemp()
    ret = subprocess.check_output("%s --version" % args.exe, shell=True)
    ret = ret.decode("utf-8")
    version = VERSION_REGEX.match(ret)
    if version is None:
        raise Exception("Failed to figure out clang-format version!")
    version = version.group(1)
    if version != EXPECTED_VERSION:
        raise Exception(
            "clang-format exe must be v%s found '%s'"
            % (EXPECTED_VERSION, version)
        )
    if len(args.dirs) == 0:
        args.dirs = DEFAULT_DIRS
    return args


def list_all_src_files(file_regex, ignore_regex, srcdirs, dstdir, inplace):
    allFiles = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(file_regex, f):
                    src = os.path.join(root, f)
                    if re.search(ignore_regex, src):
                        continue
                    if inplace:
                        _dir = root
                    else:
                        _dir = os.path.join(dstdir, root)
                    dst = os.path.join(_dir, f)
                    allFiles.append((src, dst))
    return allFiles


def run_clang_format(src, dst, exe, verbose, inplace):
    dstdir = os.path.dirname(dst)
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    # run the clang format command itself
    if src == dst:
        cmd = "%s -i %s" % (exe, src)
    else:
        cmd = "%s %s > %s" % (exe, src, dst)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print("Failed to run clang-format! Maybe your env is not proper?")
        raise
    # run the diff to check if there are any formatting issues
    if inplace:
        cmd = "diff -q %s %s >/dev/null" % (src, dst)
    else:
        cmd = "diff %s %s" % (src, dst)

    try:
        subprocess.check_call(cmd, shell=True)
        if verbose:
            print("%s passed" % os.path.basename(src))
    except subprocess.CalledProcessError:
        print(
            "%s failed! 'diff %s %s' will show formatting violations!"
            % (os.path.basename(src), src, dst)
        )
        return False
    return True


def main():
    args = parse_args()
    # Attempt to making sure that we run this script from root of repo always
    if not os.path.exists(".git"):
        print("Error!! This needs to always be run from the root of repo")
        sys.exit(-1)
    all_files = list_all_src_files(
        args.regex_compiled,
        args.ignore_compiled,
        args.dirs,
        args.dstdir,
        args.inplace,
    )
    # actual format checker
    status = True
    for src, dst in all_files:
        if not run_clang_format(
            src, dst, args.exe, args.verbose, args.inplace
        ):
            status = False
    if not status:
        print("clang-format failed! You have 2 options:")
        print(" 1. Look at formatting differences above and fix them manually")
        print(" 2. Or run the below command to bulk-fix all these at once")
        print("Bulk-fix command: ")
        print(
            "  python cpp/scripts/run-clang-format.py %s -inplace"
            % " ".join(sys.argv[1:])
        )
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
