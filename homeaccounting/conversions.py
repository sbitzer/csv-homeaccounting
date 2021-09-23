#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:46:28 2016

@author: bitzer
"""
import pathlib
import re
from warnings import warn
import functools
import time
import collections

import numpy as np
import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange as avForeignExchange
from alpha_vantage.timeseries import TimeSeries as avTimeSeries

from . import stockprices


def wait(call):
    @functools.wraps(call)
    def delayed_call(*args, **kwargs):
        out = None
        while out is None:
            try:
                out = call(*args, **kwargs)
            except ValueError as err:
                if 'API call frequency' in str(err):
                    warn('Waiting 1 min for API access.', UserWarning)
                    time.sleep(60)
                else:
                    raise
        return out

    return delayed_call


class ForeignExchange(avForeignExchange):
    @wait
    def get_currency_exchange_rate(self, from_symbol, to_symbol):
        return super().get_currency_exchange_rate(from_symbol, to_symbol)


class TimeSeries(avTimeSeries):
    @wait
    def get_symbol_search(self, keywords):
        return super().get_symbol_search(keywords)

    @wait
    def get_quote_endpoint(self, symbol):
        return super().get_quote_endpoint(symbol)


with open(pathlib.Path(__file__).parent / 'alpha_vantage.key', 'r') as file:
    avkey = file.readline()

symbol_re = re.compile(r'([A-Z\d]+)\.?([A-Z]+)?')
isin_re = re.compile(r'[a-z]{2}[a-z0-9]{9}\d$', re.IGNORECASE)
common_currencies = ['EUR', 'USD', 'GBP', 'CAD', 'CNY', 'CHF', 'DKK', 'INR',
                     'NOK', 'RUB', 'JPY', 'BRL', 'KRW']

currency_map = {'DE': 'EUR', None: 'USD'}


def search_symbol(keywords):
    ts = TimeSeries(avkey, output_format='pandas')
    info, _ = ts.get_symbol_search(keywords)

    return info


def validate_symbol(symbol):
    # 'GBX' turns up as currency for some stocks listed on the London
    # stock exchange in AlphaVantage, but AlphaVantage doesn't handle
    # 'GBX' as a currency symbol in ForeignExchange
    if symbol == 'GBX':
        symbol = 'GBP'

    return symbol

def stockprice(symbol):
    if isin_re.match(symbol):
        return stockprices.from_isin(symbol)

    ts = TimeSeries(avkey, output_format='pandas')
    info, _ = ts.get_symbol_search(symbol)
    if not info.empty:
        currency = validate_symbol(info.iloc[0]['8. currency'])
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
    elif groups[0] in common_currencies:
        return 'curr'
    elif groups[0] in ['BTC', 'XBT', 'ETH', 'ZEC', 'BCH', 'FIL']:
        return 'crypto'
    else:
        return 'stock'


class CurrencyConverter(collections.namedtuple(
        'CurrencyConverter', ['fromsym', 'tosym', 'kind', 'rate',
                              'intermediary'])):
    """

    Attributes
    ----------
    kind : {'curr2curr', 'stock2curr', 'curr2stock'}
        Description of the symbol pair: currency to currency, stock to
        currency, or currency to stock. Used internally.
    """

    def __new__(cls, fromsym, tosym, rate=None, intermediary=None):
        """Checks inputs, downloads conversion rates or prices."""
        fromsym = validate_symbol(fromsym)
        tosym = validate_symbol(tosym)

        kind = f'{identify_symbol(fromsym)}2{identify_symbol(tosym)}'

        if fromsym == tosym:
            return super(CurrencyConverter, cls).__new__(
                cls, fromsym, tosym, kind, 1, None)

        if rate is not None:
            return super(CurrencyConverter, cls).__new__(
                cls, fromsym, tosym, kind, rate, intermediary)

        rate = np.nan
        if kind in ('curr2curr', 'crypto2curr',
                    'crypto2crypto'):
            fx = ForeignExchange(key=avkey)
            out, _ = fx.get_currency_exchange_rate(fromsym, tosym)
            rate = float(out['5. Exchange Rate'])

        elif kind == 'curr2crypto':
            fx = ForeignExchange(key=avkey)
            out, _ = fx.get_currency_exchange_rate(tosym, fromsym)
            rate = 1 / float(out['5. Exchange Rate'])

        elif kind in ('stock2curr', 'curr2stock'):
            if kind == 'curr2stock':
                conv = CurrencyConverter(tosym, fromsym, intermediary)
                rate = 1 / conv.rate
            else:
                rate, currency = stockprice(fromsym)
                if currency != tosym:
                    if (intermediary and currency == intermediary.fromsym
                            and tosym == intermediary.tosym):
                        rate *= intermediary.rate
                    else:
                        warn(f"Missing intermediary for {fromsym} -> {tosym}! "
                             "Using NaN as rate!")
                        rate = np.nan

        else:
            warn(f"Unsupported conversion: {kind}! Using NaN as rate!")

        return super(CurrencyConverter, cls).__new__(
            cls, fromsym, tosym, kind, rate, intermediary)

    def __call__(self, amount):
        """Converts given amount in from-symbol to to-symbol."""
        return amount * self.rate

    def inv(self, amount):
        """Converts given amount in to-symbol to from-symbol."""
        return amount / self.rate

    def update(self, recursive=False):
        """Update conversion rates or prices.

        Only updates itermediary, if recursive = True.
        Creates new CurrencyConverter.
        """
        intermediary = self.intermediary
        if recursive:
            intermediary = intermediary.update()

        return CurrencyConverter(self.fromsym, self.tosym, intermediary)

class ConversionSet():
    """Holds all exchange rates and stock prices to be able to convert between
       all assets in a depot based on its base currency."""

    def __init__(self, depot):
        """extracts assets/symbols from accounts in depot and finds their
        conversions to the base currency."""
        self.base = depot.currency
        self.converters = {}

        accounts = depot.account_infos.copy()
        accounts.loc[:, 'symtype'] = accounts.currency.map(identify_symbol)
        currencies = accounts[
            (accounts.symtype == 'curr') | (accounts.symtype == 'crypto')]

        # first currencies, because they can be converted directly and
        # may be used as intermediaries
        for currency in (set(currencies.currency) - {self.base}):
            self.add(currency)

        # then the rest
        for currency in accounts.currency:
            self.add(currency)

    def __call__(self, currency, amount):
        if currency == self.base:
            return amount

        kind = (currency, self.base)
        if kind not in self.converters:
            self.add(currency)

        return self.converters[kind](amount)

    def add(self, currency):
        if currency == self.base or (currency, self.base) in self.converters:
            return

        from_type = identify_symbol(currency)
        if from_type in ('curr', 'crypto'):
            self.converters[(currency, self.base)] = CurrencyConverter(
                currency, self.base
            )
        elif from_type == 'stock':
            try:
                rate, stock_currency = stockprice(currency)
            except ValueError:
                self.converters[(currency, self.base)] = CurrencyConverter(
                    currency, self.base, np.nan
                )
            else:
                self.converters[(currency, stock_currency)] = (
                    CurrencyConverter(currency, stock_currency, rate)
                )
                if stock_currency != self.base:
                    self.add(stock_currency)
                    intermediary = self.converters[(stock_currency, self.base)]
                    rate *= intermediary.rate

                    self.converters[(currency, self.base)] = (
                        CurrencyConverter(
                            currency, self.base, rate, intermediary)
                    )
        else:
            self.converters[(currency, self.base)] = CurrencyConverter(
                currency, self.base, np.nan
            )

    def update(self):
        raise NotImplementedError("Implementation pending!")