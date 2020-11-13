#!/usr/bin/env python
import os
import urllib3
import subprocess
from tqdm.auto import tqdm
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor

http = urllib3.PoolManager()
CHUNK = 1024 * 2

def download_file(filename, url):
    r = http.request('GET', url, preload_content=False)
    with open(filename, 'wb') as out:
        while True:
            data = r.read(CHUNK)
            if not data: break
            out.write(data)
    r.release_conn()
    
    subprocess.call(['unzip', filename, '-d', os.path.splitext(filename)[0]])

def _namer(args):
    number, name = args
    return f'{name}_{number}.zip'

amphetamine_urls = [
    "https://www.dropbox.com/sh/gnx4htuvmb7ue65/AACB-twQSGiERqkHHEUiLc5ya?dl=1",
    "https://www.dropbox.com/sh/e2o8v9skrb0ef6b/AAA0y6no4eNPHzqMxd-eQqYxa?dl=1",
    "https://www.dropbox.com/sh/k7ieoducleexbpq/AAAQKOAjKt1k1nU2L5OzYsNsa?dl=1",
    "https://www.dropbox.com/sh/yywypn9m6z1lpyt/AABAPaoEJugycnLNI0ICNnnXa?dl=1",
    "https://www.dropbox.com/sh/0y39kqswphvemuq/AADfkxAEq5tJtwZIY8YgqNaza?dl=1",
    "https://www.dropbox.com/sh/vrqvg575ak84mjn/AAAd7lWMcawniKuuV_uHgRqga?dl=1",
    "https://www.dropbox.com/sh/4w32uvmuyrenbkp/AADGdnkEI3X5C6TTP3Jcv7mFa?dl=1",
    "https://www.dropbox.com/sh/usw26uj3cbtn2dp/AADLkV03ekUu-DVTG8-Kj14fa?dl=1",
    "https://www.dropbox.com/sh/iw17wldhuljnjk4/AAAlvlKbw9MfDFr0S5jnrGE3a?dl=1",
    "https://www.dropbox.com/sh/k4bpz38cah38kfe/AAAum45yLdrAvc-2ct6WiU59a?dl=1",
]

saline_urls = [
    "https://www.dropbox.com/sh/ca9nbj2vlsn8hyq/AACuBEODDwaizuITN6DjXv39a?dl=1",
    "https://www.dropbox.com/sh/87yzfv07d6tcrre/AAB0qXcgzGf-vRoTove34GGRa?dl=1",
    "https://www.dropbox.com/sh/u3hhekkzeejk0a5/AAAjt2WNDbX3U-G80IVDkahBa?dl=1",
    "https://www.dropbox.com/sh/a71pzlze1sbvuq3/AADTKZ_9LAzrQ3xSeJk_1GdMa?dl=1",
    "https://www.dropbox.com/sh/91gsub2ok6thnzo/AADSKuQuKSjQEcuT1SDlsICja?dl=1",
    "https://www.dropbox.com/sh/ogolgcyg7wvr9sj/AAD6Mwni4rMgsnDvnmst626na?dl=1",
    "https://www.dropbox.com/sh/kpfec602vq8tt2x/AAC7A1H9gO-I-BxjoU1pXT9la?dl=1",
    "https://www.dropbox.com/sh/9ghc0ksm7s2bip1/AABPyyA7wsZSjybHh0OjbAf6a?dl=1",
    "https://www.dropbox.com/sh/j08a0do8tresa4r/AADXQ6SdUI1Ux6DZNzr2tOBVa?dl=1",
    "https://www.dropbox.com/sh/o4j4iznywl7fkwq/AABvP6U7EEc6VqsdH8rDgpk6a?dl=1",
]

os.makedirs('small_dataset', exist_ok=True)

files = list(zip(map(_namer, enumerate(cycle(['small_dataset/amphetamine_example']))), amphetamine_urls))
files += list(zip(map(_namer, enumerate(cycle(['small_dataset/saline_example']))), saline_urls))

with ThreadPoolExecutor(3) as pool:
    for res in tqdm(pool.map(lambda args: download_file(*args), files), total=len(files)):
        pass
