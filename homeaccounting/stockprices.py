import requests

"""
Here I directly use the internal web-API of the [ING] website to get current
stock prices. The trick here is to figure out the kind of requests and their
responses that the website uses. This must be done using network analysis
of the browser (Firefox) developer tools. This directly gives you the
html request you have to send. See

https://betterprogramming.pub/how-to-scrape-modern-websites-without-headless-browsers-d871bbd1119e
"""

def from_isin(isin):
    """Uses the top search field of ING.de to look up price given ISIN.

    Looking up the price for a given ISIN works fine with this API, but
    it's not good for looking up ISINs given a name.
    """
    try:
        res = requests.get(
            'https://api.wertpapiere.ing.de/suche-autocomplete/autocomplete',
            params={'query': isin}).json()
    except requests.RequestException as err:
        raise RuntimeError(
            "Couldn't connect to source of stock prices!") from err

    R = res['total']

    curr = 'EUR'

    notfounderr = ValueError(f"No stock with ISIN '{isin}' could be found!")

    if R:
        for restype in res['suggestion_types']:
            if restype['type'] == 'direct_hit':
                for grp in restype['suggestion_groups']:
                    for hit in grp['suggestions']:
                        if hit['isin'] == isin:
                            if 'price' in hit:
                                return hit['price'], curr

    raise notfounderr

