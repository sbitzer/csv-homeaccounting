# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:38:18 2016

@author: bitzer
"""

from abc import ABCMeta, abstractmethod
import csv
import os.path
from os import scandir
import pandas as pd
import datetime as dt
import calendar
from warnings import warn

class account(metaclass=ABCMeta):
    
    def __init__(self, name='base_account', filename=None, path='.', 
                 check_for_duplicates=True, currency='EUR'):
        self.name = name
        """Account name, can contain spaces."""
        
        self.path = path
        """Path to account-specific files."""
        
        self.check_for_duplicates = check_for_duplicates
        """Whether new transactions should only be added, when they are not in the list yet."""
        
        self.filename = filename
        """File name that sould be used to store account specific files.
        
        Two account specific files will be created: 
            filename.csv containing the transaction list
            filename_transaction_files.csv containing the list of csv-files
                that were previously processed
                
        Default: Account name where spaces are replaced with underscores and 
                 all characters are lower case.
        """
        if filename is None:
            # make filename from name by replacing spaces with underscores and 
            # all lower characeters
            self.filename = self.name.replace(' ', '_').lower()
            
        self.load_transaction_files()
 
        self.balance = pd.np.nan
        """Account balance is the sum of all transaction amounts."""
        
        self.load_transactions()
        
        self.currency = currency
        

    def load_transactions(self):
        """Load previously processed and new transactions, store as DataFrame.
        
        Core function for reading transactions from csv-files. Loads previously
        processed transactions from account-specific file and checks whether 
        there are any new csv-files by comparing file names with the stored 
        list of previously processed files. Transactions found in the new files
        will be added to the transaction list. By default, it is first checked
        whether there already is an equal transaction in the list, then the 
        corresponding new transaction is ignored (see check_for_duplicates). 
        If there were new csv-files and/or transactions, the account files are 
        updated accordingly.
        """
        fullpath_all = os.path.join(self.path, self.filename + '.csv')
        
        # load previously processed transactions, if exist
        if os.path.isfile(fullpath_all):
            self.transactions = pd.read_csv(fullpath_all, index_col=0, 
                                            parse_dates=[1, 2])
        else:
            self.transactions = pd.DataFrame(columns=('booking date', 
                'value date', 'agent', 'type', 'description', 'amount'))
            
        # add transactions from new transaction files
        N_new = 0
        there_were_new_transaction_files = False
        for entry in scandir(self.path):
            if (not entry.name.startswith('.') and 
                not entry.name.startswith(self.filename) and
                entry.name.endswith('.csv') and 
                entry.is_file() and
                not entry.name in self.transaction_files):
                
                there_were_new_transaction_files = True
                self.transaction_files.append(entry.name)
                
                print("reading new csv-file: " + entry.name)
                
                new_transactions = self.get_transactions_from_csv(
                    os.path.join(self.path, entry.name))
                if self.check_for_duplicates:
                    new_transactions = self.remove_duplicate_transactions(
                        new_transactions)
                
                n_new = len(new_transactions)
                print("adding %d new transactions\n" % n_new)

                if n_new > 0:
                    N_new += n_new
                    self.transactions = self.transactions.append(new_transactions, 
                        ignore_index=True)
                
        # update balance
        self.balance = self.transactions['amount'].sum()
                
        if there_were_new_transaction_files:
            # save new file name list
            fullpath = os.path.join(self.path, self.filename + 
                '_transaction_files.csv')
                
            with open(fullpath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.transaction_files)
        
        if N_new > 0:
            # save new transaction list
            self.transactions.to_csv(fullpath_all)
            
            print('Added %d new transactions.' % N_new)


    def remove_duplicate_transactions(self, new_transactions):
        """Removes transactions from new_transactions that were already processed.
        
        Goes through new_transactions, checks whether they already exist in the
        transaction list of the account and returns only those new_transactions
        which are not in the transaction list yet.
        """
        new_inds = []
        for row in new_transactions.itertuples():
            if not (are_equal_or_nan(self.transactions['booking date'], row._1) &
                    are_equal_or_nan(self.transactions['description'], row.description) &
                    are_equal_or_nan(self.transactions['amount'], row.amount)).any():
                new_inds.append(row.Index)
                
        return new_transactions.loc[new_inds]

    
    @abstractmethod
    def get_transactions_from_csv(self, filepath):
        """read new transactions from account-specific csv-file"""
        pass


    def load_transaction_files(self):
        """load the list of already processed transaction file names"""
        fullpath = os.path.join(self.path, self.filename + 
            '_transaction_files.csv')
        if os.path.isfile(fullpath):
            with open(fullpath, 'r', newline='') as file:
                reader = csv.reader(file)
                for filenames in reader:
                    pass
            self.transaction_files = filenames
        else:
            self.transaction_files = []
            
            
    def __str__(self):
        return self.name + ': %8.2f %s' % (self.balance, self.currency)
        
    
    def get_month(self, month_id=0, datetype='value date'):
        """Get transactions from a given month.
        
        month_id may be an integer, then it is the month_id-th month before the 
                 current one (both negative and positive have the same effect)
                 may also be a date object or a string in the form yyyy-mm
        datetype is either 'value date' (default) or 'booking date'
                 determines which type of date is used to select transactions
                 
        returns a DataFrame with the corresponding transactions
        """
        
        if type(month_id) == int:
            date = dt.date.today()
            for m_it in range(abs(month_id)):
                # get a day in the previous month
                date = date.replace(day=1) - dt.timedelta(1)
        else:
            date = parse_date(month_id)
            
        first_day = date.replace(day=1)
        last_day = date.replace(day=calendar.monthrange(date.year, date.month)[1])
        
        tr = self.transactions[(self.transactions[datetype] >= first_day) & 
                               (self.transactions[datetype] <= last_day)]
        
        return tr.sort_values(datetype)
        
        
    def get_last(self, N=3, datetype='value date'):
        tr = self.transactions.sort_values(datetype, ascending=False)
        
        return tr.iloc[:min(N, len(tr))]
                       
        
class ing_diba_giro(account):
    """An implementation of account specific for ING DiBa currency accounts in Germany."""
    
    def __init__(self, name='ING DiBa Giro', **account_args):
        super().__init__(name=name, **account_args)
        
        
    def get_transactions_from_csv(self, filepath):
        """Reads csv-files downloaded from German ING DiBa online banking."""
        
        # open the file and read lines until finding the one which starts with 
        # "Buchung" this should be the number of rows to skip (there can be 8, 
        # or 10)
        with open(filepath, encoding='latin_1') as file:
            skiprows = 0
            for line in file:
                skiprows += 1
                if line.startswith(('"Buchung";', 'Buchung;')):
                    break
                
        """
        The options here define the csv-dialect of ING DiBa csv-files 
        downloaded from the German branch of the bank. The first few lines not
        containing transactions are skipped. The transaction columns are 
        associated with the appropriate columns of the account-DataFrame. Dates
        and transaction amounts are in German-style and handled accordingly.
        The text fields agent and description are stripped of unnecessary 
        whitespace and return nan, if empty.
        """
        return pd.read_csv(filepath, sep=';', header=None, skiprows=skiprows, 
            names=('booking date', 'value date', 'agent', 'type', 
            'description', 'amount'), usecols=range(6), skipinitialspace=True, 
            decimal=',', thousands='.', parse_dates=[0, 1], dayfirst=True,
            converters={'agent': rmspace, 'description': rmspace},
            encoding='latin_1')

            
class manual_current(account):
    """A manually maintained account using the basic csv-format as follows:
    
    1 index column + 6 typical account columns
    1st row with column names
    comma-separation
    dates (2nd and 3rd columns) in format yyyy-mm-dd
    """
    
    def __init__(self, name='default manual', **account_args):
        super().__init__(name=name, **account_args)
    
        
    def load_transactions(self):
        """Load previously processed transactions, store as DataFrame."""
        
        fullpath_all = os.path.join(self.path, self.filename + '.csv')
        
        # load previously processed transactions, if exist
        if os.path.isfile(fullpath_all):
            self.transactions = pd.read_csv(fullpath_all, index_col=0, 
                                            parse_dates=[1, 2])
        else:
            self.transactions = pd.DataFrame(columns=('booking date', 
                'value date', 'agent', 'type', 'description', 'amount'))
            
        # update balance
        self.balance = self.transactions['amount'].sum()
        
        
    def get_transactions_from_csv(self, filepath):
        """As this is a manual account no transactions will be loaded."""
        return pd.DataFrame(columns=('booking date', 'value date', 'agent', 
                                     'type', 'description', 'amount'))
        
    def add_transaction(self, vdate, agent, desc, amount, bdate=None, 
                        t_type=''):
        """Add a transaction manually to the account."""
        
        # check format of vdate
        vdate = parse_date(vdate)
        
        if bdate is None:
            bdate = vdate
        else:
            bdate = parse_date(bdate)
            
        transaction = pd.DataFrame([[bdate, vdate, rmspace(agent), 
            rmspace(t_type), rmspace(desc), float(amount)]], columns=(
            'booking date', 'value date', 'agent', 'type', 'description', 
            'amount'))
        
        if self.check_for_duplicates:
            transaction = self.remove_duplicate_transactions(transaction)
            
        if len(transaction) > 0:
            self.transactions = self.transactions.append(transaction, 
                                                         ignore_index=True)
            # update balance
            self.balance = self.transactions['amount'].sum()
            
            self.transactions.to_csv(os.path.join(self.path, self.filename + 
                                                  '.csv'))
        else:
            warn('Transaction already exists! Did not add.')
                
            
def rmspace(s):
    """Removes extra whitespace and returns nan, if empty string."""
    s = " ".join(s.split())
    if len(s) > 0:
        return s
    else:
        return pd.np.nan
        
        
def are_equal_or_nan(array, val):
    """checks whether val is in array, including when val is nan"""
    return (array == val) | ((array != array) & (val != val))
    
def parse_date(date):
    """Parse the date into pandas Timestamp, handle German dd.mm.yyyy manually"""
    if type(date) is not pd.Timestamp:
        if isinstance(date, str) and date[2] == '.':
            date = pd.Timestamp(pd.datetime.strptime(date, '%d.%m.%Y'))
        else:
            date = pd.Timestamp(date)
                    
    return date