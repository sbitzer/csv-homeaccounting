#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:27:08 2017

@author: bitzer
"""

import os
from glob import iglob

os.remove(os.path.join('data', 'ing-diba', 'ing-diba.csv'))
os.remove(os.path.join('data', 'ing-diba', 'ing-diba_transaction_files.csv'))
os.remove(os.path.join('data', 'demo_depot.csv'))
for macc in iglob(os.path.join('data', 'manual_accounts', '*.csv')):
    os.remove(macc)

