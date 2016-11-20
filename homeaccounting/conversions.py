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

symbol_re = re.compile(r'([A-Z]+)\.?([A-Z]+)?')

currency_map = {'DE': 'EUR'}

def convert(x, from_symbol, to_symbol):
    from_symbol = from_symbol.upper()
    to_symbol = to_symbol.upper()
    
    from_match = symbol_re.match(from_symbol)
    to_match = symbol_re.match(to_symbol)
    
    y = None
    if from_symbol == to_symbol:
        y = x
    elif from_match is None:
        warn('converter: did not recognise from_symbol "%s", returning NaN!' % 
             from_symbol)
        return np.nan
    elif to_match is None:
        warn('converter: did not recognise to_symbol "%s", returning NaN!' % 
             to_symbol)
        return np.nan
    elif to_match.groups()[1] is not None:
        raise ValueError('Conversion to stocks currently not supported!')
    else:
        from_groups = from_match.groups()
        
        # from bitcoin to currency
        if from_groups[0] in ['BTC', 'XBT']:
            bconv = fx.bitcoin.BtcConverter()
            y = bconv.convert_btc_to_cur(x, to_symbol)
        
        # from something to bitcoin
        elif to_symbol in ['BTC', 'XBT']:
            if from_groups[1] is not None:
                x = convert(x, from_symbol, 'EUR')
                from_symbol = 'EUR'
                
            bconv = fx.bitcoin.BtcConverter()
            y = bconv.convert_to_btc(x, from_symbol)
        
        # from stock to something
        elif from_groups[1] is not None:
            stock_info = ystockquote.get_all(from_symbol)
            price = convert(float(stock_info['price']), 
                            currency_map[from_groups[1]], to_symbol)
            y = x * price
        
        # from currency to currency
        else:
            cconv = fx.converter.CurrencyRates()
            y = cconv.convert(from_symbol, to_symbol, x)
            
    if y is None:
        y = np.nan
        warn('converter: did not recognise one of the symbols, returning NaN!')
        
    return y