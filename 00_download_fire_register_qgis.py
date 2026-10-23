"""
Download the RSDI Basilicata burned-area layers (aree percorse dal fuoco) as GeoJSON,
one file per year, with a plain WFS GetFeature request (no QGIS layers or Processing involved).

HOW TO RUN: QGIS -> Plugins -> Python Console -> Show Editor -> Open this file -> Run script.
(It also runs with any normal Python 3: python scarica_incendi_wfs_v3.py)

OUTPUT: <OUT_DIR>\incendi_geojson\incendi_2016.geojson ... incendi_2025.geojson (EPSG:4326)
"""
import os, json, urllib.request, urllib.parse, ssl

WFS_URL = 'https://rsdi.regione.basilicata.it/rbgeoserver2016/aree_percorse_dal_fuoco/wfs'
WORKSPACE = 'aree_percorse_dal_fuoco'
YEARS = range(2016, 2026)
OUT_DIR = os.path.join(os.path.expanduser('~'), 'basilicata_fire_register_geojson')   # edit if needed
os.makedirs(OUT_DIR, exist_ok=True)


def get(params):
    url = WFS_URL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()
    except ssl.SSLError:
        ctx = ssl._create_unverified_context()          # some regional servers have incomplete certificate chains
        with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
            return r.read()


summary = []
for y in YEARS:
    layer = f'incendi_{y}'
    ok = False
    for typename in (f'{WORKSPACE}:{layer}', layer):
        params = {'service': 'WFS', 'version': '1.0.0', 'request': 'GetFeature',
                  'typeName': typename, 'outputFormat': 'application/json', 'srsName': 'EPSG:4326'}
        try:
            data = get(params)
            gj = json.loads(data)
            n = len(gj.get('features', []))
            path = os.path.join(OUT_DIR, f'{layer}.geojson')
            with open(path, 'wb') as f:
                f.write(data)
            print(f'{layer}: {n} features saved')
            summary.append((layer, n))
            ok = True
            break
        except Exception as e:
            last = e
    if not ok:
        print(f'{layer}: FAILED ({last})')
        summary.append((layer, None))

print('DONE ->', OUT_DIR)
