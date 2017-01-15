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
import matplotlib.pyplot as plt

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


    def remove_transactions(self, ids):
        """remove transactions by id
        
        Is useful when you added a wrong transaction manually. Updates the
        csv-file.
        
        Arguments
        ---------
        ids : int or iterable of ints
            the ids of the transactions to be removed
            
        Return
        ------
        removed transactions : DataFrame
        """
        tr = self.transactions.loc[ids]
        self.transactions.drop(ids, inplace=True)
        self.transactions.to_csv(os.path.join(self.path, self.filename + '.csv'))
        
        print('Removed transactions:')
        return tr
        
            
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
            # parse_date returns a pandas timestamp - convert to date
            date = date.date()
            
        first_day = date.replace(day=1)
        last_day = date.replace(day=calendar.monthrange(date.year, date.month)[1])
        
        tr = self.transactions[(self.transactions[datetype] >= first_day) & 
                               (self.transactions[datetype] <= last_day)]
        
        return tr.sort_values(datetype), last_day
        
        
    def get_last(self, N=3, datetype='value date'):
        """Get last N transactions."""
        
        tr = self.transactions.sort_values(datetype, ascending=False)
        
        return tr.iloc[:min(N, len(tr))]
                       
                       
    def get_monthly_balance(self, months=0, datetype='value date'):
        """Gives summary of transactions in selected months.
        
        months is either a single month_id as in get_month, or an iterable over
               such ids
        
        datetype is the type of date that is used to select transactions for 
                 the corresponding month, see get_month
                 
        returns a pandas DataFrame with the last day of the selected month as
                index, each row consists of 'debit' (total amount of account
                debits), 'credit' (total, negative amount of account credits), 
                'balance' (debit + credit)
        """
        if not hasattr(months, '__iter__') or type(months) == str:
            months = [months]

        days = []
        balances = []
        for month in months:
            tr, day = self.get_month(month, datetype=datetype)
            days.append(day)
            amounts = tr['amount']
            balances.append([amounts[amounts>0].sum(), 
                             amounts[amounts<0].sum(),
                             amounts.sum()])
            
        df = pd.DataFrame(balances, columns=['debit', 'credit', 'balance'],
                          index=days)
        
        return df
        
        
    def find(self, bdate=None, vdate=None, agent=None, t_type=None, 
             description=None, amount=None, case=False):
        """Find transactions matching flexible criteria.
        
        Each column of the transaction DataFrame has its corresponding keyword
        argument here. Transactions are returned whose values match those given
        to the function. 
        
        Dates (bdate and vdate) can be given in the usual formats (e.g. 
        '20141115', '15.11.2014', or as any datetime-derived object). 
        
        The string columns will be searched for strings which contain the given
        search string. Default is case insensitive search. Change this by 
        setting case=True.
        """
        
        if bdate is not None:
            bdate = parse_date(bdate)
            
        if vdate is not None:
            vdate = parse_date(vdate)
            
        if agent is None:
            a_match = pd.np.ones(len(self.transactions), dtype=bool)
        else:
            a_match = self.transactions['agent'].str.contains(agent, case)
            
        if t_type is None:
            t_match = pd.np.ones(len(self.transactions), dtype=bool)
        else:
            t_match = self.transactions['type'].str.contains(t_type, case)
            
        if description is None:
            d_match = pd.np.ones(len(self.transactions), dtype=bool)
        else:
            d_match = self.transactions['description'].str.contains(description, case)

        return self.transactions[
            are_equal_or_nan(self.transactions['booking date'], bdate) &
            are_equal_or_nan(self.transactions['value date'], vdate) & 
            a_match & t_match & d_match & 
            are_equal_or_nan(self.transactions['amount'], amount)]

            
    def plot_history(self, what='balance', startdate=None, enddate=None, 
                     datetype='value date'):
        """Visualises the transaction history.
        
        Arguments
        ---------
        what : str, default 'balance'
            what exactly should be plotted; 
            'balance': balance of the account at the specified dates 
            'amount':  individual transaction amounts
            'cumsum':  cumulative sum of the transaction amounts in the 
                       selected time period; equals balance when the startdate
                       is equal to or before the date of the first transaction
        
        startdate : date (str or object), default None
            start date from which transactions should be shown, will be parsed
            into date format
            
        enddate : date (str or object), default None
            last date from which transactions should be shown, will be parsed
            into date format
            
        datetype : str, default 'value date'
            either 'booking date' or 'value date'
            
        Returns
        -------
        matplotlib figure instance
        """
        if startdate is None:
            startdate = self.transactions[datetype].min()
        else:
            startdate = parse_date(startdate)
        if enddate is None:
            enddate = pd.datetime.today()
        else:
            enddate = parse_date(enddate)
        
        tr = self.transactions
        tr = tr.sort_values(datetype)
        
        if what == 'balance':
            tr['balance'] = tr['amount'].cumsum()
            
        tr = tr[(tr[datetype] >= startdate) & (tr[datetype] <= enddate)]
        
        if what == 'cumsum':
            tr['cumsum'] = tr['amount'].cumsum()
        
        fig = plt.figure()
        plt.plot(tr[datetype], tr[what])
        fig.autofmt_xdate()
        plt.ylabel(what)
        
        return fig
        
        
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
                

class credit_subaccount():
    """Keeps track of a credit given to someone based on transactions in an account.
    
        Uses search terms to find transactions modifying the credit balance and
        computes daily interests rates for a given annual equivalent rate.
    """
    
    def __init__(self, acc, desc, aer, tr_inds=pd.np.array([]), agent=None,
                 verbose=True):
        """Define a credit account based on a standard current account.
        
            The credit account is solely based on selected transactions of the
            associated current account which can either be given as index into
            the transaction list, or using search terms in the description 
            and/or agent of the transactions.
            
            Arguments
            ---------
            acc : account
                the associated current account.
                
            desc : string
                search pattern for description of transactions.
                
            aer : number (percent)
                interest rate to be payable, as `annual equivalent rate`_
            
            tr_inds : 1D numpy array, default empty array
                indeces into transaction list of acc, pointing to transactions
                which cannot be identified by the search patterns
                
            agent : string, default None
                search pattern for agent of transactions, will be combined with
                search pattern for description
                
            verbose : bool, default True
                whether to print state of credit account upon creation
                
            Returns
            -------
            instance of credit_subaccount defined by arguments
                
            .. _annual equivalent rate: https://en.wikipedia.org/wiki/Effective_interest_rate
        """
        self.acc = acc
        self.desc = desc
        self.agent = agent
        self.tr_inds = tr_inds
        self.aer = aer
        
        if verbose:
            print(self.__str__())
        
    def get_transactions(self):
        """Returns transactions associated with the credit, sorted by value date."""
        
        tr = self.acc.find(description=self.desc, agent=self.agent)
        inds = pd.np.unique(pd.np.r_[self.tr_inds, tr.index])
        return self.acc.transactions.loc[inds].sort_values('value date')
        
    def get_state(self):
        """Calculate and return current state of credit account.
        
            Calculates current balance of credit account including accrued 
            interest and already made back payments. Interest is calculated on
            a daily basis for all intermediate balances.
        
            Returns
            -------
            balance
                outstanding credit including interest that still needs to be 
                payed back
            
            credit
                total given credit without interest and already made payments
                
            interest
                total interest accrued until today
        """
        tr = self.get_transactions()
        
        lastdate = tr.iloc[0]['value date']
        credit = 0
        balance = 0
        interest = 0
        for row in tr.itertuples():
            # compute interest between last value date and current value date
            dt = (row._2 - lastdate).days
            lastdate = row._2
            factor = (1 + self.aer /100 / 365) ** dt
            interest += (factor - 1) * balance
            
            # update balance with computed interest
            balance *= factor
            
            # update balance with new transaction
            balance += row.amount
            
            # save credit
            credit += pd.np.min([row.amount, 0])
            
        # add interest from last transaction to today
        dt = (pd.datetime.today() - lastdate).days
        factor = (1 + self.aer / 100 / 365) ** dt
        interest += (factor - 1) * balance
        balance *= factor
        
        return balance, credit, interest
    
    def __str__(self):
        balance, credit, interest = self.get_state()
        
        s =  'current balance:    %+8.2f\n' % balance
        s += 'total given credit: %+8.2f\n' % credit
        s += 'accrued interest:   %+8.2f' % interest
        
        return s

        
def rmspace(s):
    """Removes extra whitespace and returns nan, if empty string."""
    s = " ".join(s.split())
    if len(s) > 0:
        return s
    else:
        return pd.np.nan
        
        
def are_equal_or_nan(array, val):
    """Checks whether val is in array, including when val is nan.
    
    If val is None, it's treated as a wildcard which is matched by all elements
    in array.
    """
    
    if val is None:
        return pd.np.ones(len(array), dtype=bool)
    else:
        return (array == val) | ((array != array) & (val != val))
    
        
def parse_date(date):
    """Parse the date into pandas Timestamp, handle German dd.mm.yyyy manually"""
    if type(date) is not pd.Timestamp:
        if isinstance(date, str) and date[2] == '.':
            date = pd.Timestamp(pd.datetime.strptime(date, '%d.%m.%Y'))
        else:
            date = pd.Timestamp(date)
                    
    return date