""" Module to give helpful messages to the user that did not
compile sdeconv properly.
"""
import os

INPLACE_MSG = """
It appears that you are importing a local sdeconv source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""


def raise_build_error(err):
    """ Raise a comprehensible error and list the contents of the
     directory to help debugging on the mailing list.
    """
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == "sdeconv/__check_build":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = INPLACE_MSG
    dir_content = []
    for i, filename in enumerate(os.listdir(local_dir)):
        if (i + 1) % 3:
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + '\n')
    raise ImportError("""%s
___________________________________________________________________________
Contents of %s:
%s
___________________________________________________________________________
It seems that sdeconv has not been built correctly.

If you have installed sdeconv from source, please do not forget
to build the package before using it: run `python setup.py install` or
`make` in the source directory.
%s""" % (err, local_dir, ''.join(dir_content).strip(), msg))
