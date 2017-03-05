csv-homeaccounting
==================

A low-tech solution for keeping on top of your finances across different institutions and asset classes.


Features
---------
- **grand overview**: make an overview over all your finances across currency and 
  saving accounts, stocks and bitcoin
- **convert**: compare the value of assets, e.g., stocks and bitcoin, by automatically converting them to a common currency, e.g., EUR
- **use csv-files from banks**: keep track of bank accounts based on csv-files that 
  you can typically download from bank websites
- **loans**: keep track of loans that you give to people
- **interoperability**: csv as low-tech solution to try to stay compatible with as 
  many institutions as possible and be able to manually edit, if necessary
- **local**: everything runs on your computer and you're in control of your data


What ``csv-homeaccounting`` can't do
....................................
- buy or sell assets
- send or receive money

It's no banking package. It just collects and analyses executed transactions.


Installation
------------
Clone the repository and do

.. code-block::

    python setup.py install
    
Dependencies
............
- pandas
- forex-python_
- ystockquote_

Potential issues
................
In my installation I had to add the following line to forex-python's ``__init__.py``:

.. code-block::

    from . import converter


Usage
-----
An extensive demo of the package can be found in the jupyter notebook Homeaccounting_Demo.ipynb_ under ``examples``.


.. _forex-python: https://github.com/MicroPyramid/forex-python
.. _ystockquote: https://github.com/cgoldberg/ystockquote
.. _Homeaccounting_Demo.ipynb: examples/Homeaccounting_Demo.ipynb