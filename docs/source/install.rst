Installation
============

To install Atlas, we recommend installing from from source ::

    git clone https://github.com/aspuru-guzik-group/atlas.git
    cd atlas
    pip install -e .
    pip install -r requirements.txt

Atlas works hand-in-hand with Olympus, which can be installed from source. Specifically, the ``olympus-atlas`` branch for compatibility with Atlas. ::

    git clone olympus-atlas --single-branch https://github.com/aspuru-guzik-group/olympus.git
    cd olympus
    python install -e .

Dependencies
------------
The installation only requires:

* ``python >= 3.9``
* ``numpy``
* ``pandas``

Additional libraries are required to use specific :ref:`planners` and :ref:`emulators`. However, **Olympus** will alert
you about these requirements as you try access the related functionality.




