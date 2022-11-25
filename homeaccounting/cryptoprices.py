import requests

def check(sym):
    assert isinstance(sym, str), "Symbol name must be str!"
    if len(sym) != 3:
        raise ValueError("Can only process 3-letter symbols!")


def get_exchange_rate(fromsym, tosym, try_reverse=True):
    """Uses Kraken public ticker API to get exchange rates."""
    check(fromsym)
    check(tosym)

    resp = requests.get(
        f'https://api.kraken.com/0/public/Ticker?pair={fromsym}{tosym}').json()

    if resp['error']:
        if 'Unknown asset pair' in resp['error'][0]:
            return 1 / get_exchange_rate(tosym, fromsym, try_reverse=False)
        else:
            raise ValueError(str(resp['error']))

    return float(next(iter(resp['result'].values()))['c'][0])
