Install
=======

This section contains the instructions to install ``SDeconv``

Using PyPI
----------

Releases are available in PyPI a repository. We recommend using virtual environment

.. code-block:: shell

    python -m venv .sdeconv-env
    source .sdeconv-env/bin/activate
    pip install sdeconv


From source
-----------

If you plan to develop ``SDeconv`` we recommend installing locally

.. code-block:: shell

    python -m venv .sdeconv-env
    source .sdeconv-env/bin/activate
    git clone https://github.com/sylvainprigent/sdeconv.git
    cd sdeconv
    pip install -e .
