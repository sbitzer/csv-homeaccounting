#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:46:28 2016

@author: bitzer
"""
import pathlib
import re
from warnings import warn

import numpy as np
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries

with open(pathlib.Path(__file__).parent / 'alpha_vantage.key', 'r') as file:
    avkey = file.readline()

symbol_re = re.compile(r'([A-Z\d]+)\.?([A-Z]+)?')
common_currencies = ['EUR', 'USD', 'GBP', 'CAD', 'CNY', 'CHF', 'DKK', 'INR',
                     'NOK', 'RUB', 'JPY', 'BRL', 'KRW']

currency_map = {'DE': 'EUR', None: 'USD'}

def search_symbol(keywords):
    ts = TimeSeries(avkey)
    info, _ = ts.get_symbol_search(keywords)

    return info

def stockprice(symbol):
    ts = TimeSeries(avkey)
    info, _ = ts.get_symbol_search(symbol)
    if info:
        currency = info[0]['8. currency']
    else:
        raise ValueError(f"Symbol '{symbol}' not found in AlphaVantage!")

    out, _ = ts.get_quote_endpoint(symbol)
    if out:
        price = float(out['05. price'])
    else:
        raise ValueError(f"No data for '{symbol}' on AlphaVantage!")

    return price, currency

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
            if from_type in ['crypto', 'currency'] and to_type in ['crypto', 'currency']:
                fx = ForeignExchange(key=avkey)
                out, _ = fx.get_currency_exchange_rate(from_symbol, to_symbol)
                
                y = x * float(out['5. Exchange Rate'])
                
            elif from_type in ['crypto', 'currency'] and to_type == 'stock':
                price, currency = stockprice(to_symbol)

                if from_symbol != currency:
                    x = convert(x, from_symbol, currency)

                y = x / price
                
            elif from_type == 'stock' and to_type in ['crypto', 'currency']:
                price, currency = stockprice(from_symbol)
                if currency != to_symbol:
                    price = convert(price, currency, to_symbol)
                
                y = x * price
                
            elif from_type == 'stock' and to_type == 'stock':
                x = convert(x, from_symbol, 'EUR')
                y = convert(1, to_symbol, 'EUR')

                y = x / y
                
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
        return 'stock'
    elif len(groups[0]) == 12:
        return 'ISIN'
    elif groups[0] in common_currencies:
        return 'currency'
    elif groups[0] in ['BTC', 'XBT', 'ETH', 'ZEC', 'BCH']:
        return 'crypto'
    else:
        return 'stock'
