# download megatools
import os
import gdown

def drivedown(download_link, filename='.', verbose=False):
    """Use X binary executable to download files and folders from Google Drive ."""
    assert not os.path.exists(filename), f'gdown cannot download to already occupied location: "{filename}"'
    abs_filepath = os.path.abspath(filename)
    gdown.download(download_link, abs_filepath, quiet=not verbose)
    assert os.path.exists(filename), f'gdown downloaded file does not exist in expected output location: "{filename}"'
