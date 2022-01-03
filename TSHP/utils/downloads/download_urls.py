import os
import urllib.request
from tqdm import tqdm
import gdown
from os.path import exists

from TSHP.utils.downloads.download_mega import megadown
from TSHP.utils.downloads.extract import extract


def request_url_with_progress_bar(url, filename):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    def download_url(url, filename):
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    download_url(url, filename)


def download(urls, dataset='', filenames=None, working_dir=None, force_dl=False, username='', password='', auth_needed=False, extract_inplace=True):
    old_cwd = os.getcwd()
    if working_dir is not None:
        os.chdir(working_dir)
    if type(urls) is str:
        urls = [urls, ]
    if type(filenames) is str:
        filenames = [filenames, ]
    assert filenames is None or len(urls) == len(filenames), f"number of urls does not match filenames. Expected {len(filenames)} urls, containing the files listed below.\n{filenames}"
    assert not auth_needed or (len(username) and len(password)), f"username and password needed for {dataset} Dataset"
    if filenames is None:
        filenames = [None,]*len(urls)
    
    for i, (url, filename) in enumerate(zip(urls, filenames)):
        print(f"Downloading File {i+1}/{len(urls)} from '{urls[i]}'...")
        #if filename is None:
        #    filename = url.split("/")[-1]
        if filename and (not force_dl) and exists(filename):
            print(f"{filename} Already Exists, Skipping.")
            continue
        if 'drive.google.com' in url:
            assert 'https://drive.google.com/uc?id=' in url, 'Google Drive links should follow the format "https://drive.google.com/uc?id=1eQAnaoDBGQZldPVk-nzgYzRbcPSmnpv6".\nWhere id=XXXXXXXXXXXXXXXXX is the Google Drive Share ID.'
            gdown.download(url, filename, quiet=False)
        elif 'mega.nz' in url:
            megadown(url, filename, verbose=True)
        else:
            #urllib.request.urlretrieve(url, filename=filename) # no progress bar
            request_url_with_progress_bar(url, filename) # with progress bar
    
    if extract_inplace:
        for name in os.listdir(os.getcwd()):
            if any(name.endswith(x) for x in [".zip", ".tar.bz2", ".tar.gz", ".tar", ".7z"]):
                extract(name)
                os.remove(name)
    
    if working_dir is not None:
        os.chdir(old_cwd)
    print("Finished!")


def download_unknown(output_dir, id, site):
    """
    Download GDrive/MEGA/URL Folder/File/Zip to Dataset Path.
    Attempt to ensure files outside the directory cannot be affected by download or extraction.
    """
    cwd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    site_lower = site.lower()
    if site_lower in ['url']:
        download(id, '', None)
    else:
        raise NotImplementedError()
    os.chdir(cwd)
    return