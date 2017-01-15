#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:46:28 2016

@author: bitzer
"""

# NOTE that I had to adapt forex_python's __init__.py so that it also loads 
# converter and not only bitcoin - something must have gone wrong during install?
import forex_python as fx
import ystockquote
import numpy as np
import re
from warnings import warn

symbol_re = re.compile(r'([A-Z\d]+)\.?([A-Z]+)?')
common_currencies = ['EUR', 'USD', 'GBP', 'CAD', 'CNY', 'CHF', 'DKK', 'INR',
                     'NOK', 'RUB', 'JPY', 'BRL', 'KRW']

currency_map = {'DE': 'EUR', None: 'USD'}

def convert(x, from_symbol, to_symbol):
    from_symbol = from_symbol.upper()
    to_symbol = to_symbol.upper()
    
    from_type = identify_symbol(from_symbol)
    to_type = identify_symbol(to_symbol)
    
    y = None
    if from_symbol == to_symbol:
        return x
    
    if from_type is None:
        warn('converter: did not recognise from_symbol "%s", returning NaN!' % 
             from_symbol)
        return np.nan
    elif from_type == 'ISIN' or to_type == 'ISIN':
        warn('converter: cannot currently convert assets to or from ISIN, '
             'returning NaN!')
        return np.nan
    elif to_type is None:
        warn('converter: did not recognise to_symbol "%s", returning NaN!' % 
             to_symbol)
        return np.nan
    else:
        try:
            # from bitcoin to something
            if from_type == 'bitcoin':
                bconv = fx.bitcoin.BtcConverter()
                if to_type != 'currency':
                    x = bconv.convert_btc_to_cur(x, 'EUR')
                    y = convert(x, 'EUR', to_symbol)
                else:
                    y = bconv.convert_btc_to_cur(x, to_symbol)
                
            # from something to bitcoin
            elif to_type == 'bitcoin':
                if from_type != 'currency':
                    x = convert(x, from_symbol, 'EUR')
                    from_symbol = 'EUR'
                    
                bconv = fx.bitcoin.BtcConverter()
                y = bconv.convert_to_btc(x, from_symbol)
            
            # from stock/fund to something
            elif from_type == 'yahoo':
                from_groups = symbol_re.match(from_symbol).groups()
                stock_info = ystockquote.get_all(from_symbol)
                price = convert(float(stock_info['price']), 
                                currency_map[from_groups[1]], to_symbol)
                y = x * price
                
            # from something to stock/fund
            elif to_type == 'yahoo':
                to_groups = symbol_re.match(to_symbol).groups()
                stock_info = ystockquote.get_all(to_symbol)
                price = convert(float(stock_info['price']), 
                                currency_map[to_groups[1]], from_symbol)
                y = x / price
            
            # from currency to currency
            elif from_type == 'currency' and to_type == 'currency':
                cconv = fx.converter.CurrencyRates()
                y = cconv.convert(from_symbol, to_symbol, x)
                
        except Exception as err:
            warn('converter: unexpected error ({}) during conversion, '
                 'returning NaN!'.format(err))
            return np.nan
            
    if y is None:
        y = np.nan
        warn('converter: did not recognise one of the symbols, returning NaN!')
        
    return y
    
    
def identify_symbol(symbol):
    match = symbol_re.match(symbol)
    
    if match is None:
        return None
    
    groups = match.groups()
        
    if groups[1] is not None:
        return 'yahoo'
    elif len(groups[0]) == 12:
        return 'ISIN'
    elif groups[0] in common_currencies:
        return 'currency'
    elif groups[0] in ['BTC', 'XBT']:
        return 'bitcoin'
    else:
        return 'yahoo'