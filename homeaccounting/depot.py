#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 18:15:41 2016

@author: bitzer
"""

import os.path
import pandas as pd
import re
from . import accounts
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class depot(object):
    @property
    def accounts(self):
        return self._accounts
    
    account_info_cols = ('type', 'name', 'filename', 'path', 
                         'check_for_duplicates', 'currency')
        
    info_re = re.compile(r'^[\w\\/\.-]+$')
        
    def __init__(self, name='default depot', filename=None, path='.', 
                 currency='EUR'):
        self.name = name
        """Depot name, can contain spaces."""
        
        self.path = path
        """Path to depot-specific files."""
        
        self.filename = filename
        """File name that sould be used to store depot specific files.
        
        One depot specific file will be created: 
            filename.csv containing information about accounts associated
                with this depot
                
        Default: depot name where spaces are replaced with underscores and 
                 all characters are lower case.
        """
        if filename is None:
            # make filename from name by replacing spaces with underscores and 
            # all lower characeters
            self.filename = self.name.replace(' ', '_').lower()
            
        self.currency = currency
        """Currency in which depot is held."""
        
        self._accounts = []
        """The list of accounts held in this depot."""
        self.account_infos = None
        """Dataframe containing type and location of accounts."""
        self.load_accounts()

        
    def load_accounts(self):
        """Load the accounts previously added to this depot."""
        
        fullpath = os.path.join(self.path, self.filename + '.csv')
        if os.path.isfile(fullpath):
            self.account_infos = pd.read_csv(fullpath, index_col=0)
        else:
            self.account_infos = pd.DataFrame(columns=self.account_info_cols)
        
        for row in self.account_infos.itertuples():
            row = row._asdict()
            # first make sure that the account infos are only valid strings
            for info in self.account_infos.columns:
                if isinstance(row[info], str) and self.info_re.match(row[info]) is None:
                    raise(ValueError('Could not recognise %s of account %s, '
                                     'when loading the depot! Has the depot file '
                                     'been changed manually?' % (info, row['name'])))
            
            # now it should be safe to create the account
            self._accounts.append(eval('accounts.' + row['type'] + '(name=row["name"], '
                'filename=row["filename"], path=row["path"], '
                'check_for_duplicates=row["check_for_duplicates"], '
                'currency=row["currency"])'))
                    
                    
    def add_account(self, acc):
        """Add an account to this depot."""
        
        # check whether account already exists
        if any((self.account_infos.name == acc.name) & 
               (self.account_infos.path == acc.path)):
            print('Account already exists in depot. Doing nothing.')
            return
        
        # add new row to account infos dataframe
        self.account_infos = self.account_infos.append(pd.DataFrame([[
            acc.__class__.__name__, acc.name, acc.filename, acc.path, 
            acc.check_for_duplicates, acc.currency]], 
            columns=self.account_info_cols), ignore_index=True)
        
        self.account_infos.to_csv(os.path.join(self.path, self.filename + '.csv'))
        
        # add to account list
        self._accounts.append(acc)
        
        
    def show_overview(self):
        """Shows account balances as a pie-plot."""
        
        names = []
        balances = []

        for acc in self.accounts:
            names.append(acc.name)
            balances.append(acc.balance)
        
        numzeros = math.ceil(math.log10(max(balances)))
        total = sum(balances)
        fmt = '%'+str(numzeros+3)+'.2f'
        balfun = lambda pct: fmt % (pct / 100 * total)
        
        cols = plt.get_cmap('Accent')
        cols = cols(np.linspace(0, 1, len(names)))
        
        ax = plt.axes(aspect=1)
        ax.pie(balances, labels=names, autopct=balfun, colors=cols)
        ax.set_title('balances in ' + self.currency)