# Path handling and other niceties for easy reference and less headache
import os
import tarfile
import requests

from tqdm import tqdm

pj = os.path.join

# abspath also norms the path
data_dir = os.path.abspath(pj(os.path.dirname(__file__), '..', 'data'))

# Path configuration based on file category
FILE_CATS = {
    'RADAR_REFL_COMP': pj(data_dir, 'radar_refl_comp')
}


def download_cache(file_cat, url, filename=None, show_progress=True, extract_tar=False):
    """
    Download file or return cached, optionally extract TAR archive returning
    list of files.
    """
    try:
        file_dir = FILE_CATS[file_cat]
    except KeyError:
        raise Warning("File category '%s' was not configured", file_cat)
    
    # Create parent dir if needed
    if not os.path.exists(file_dir):
        print("Creating directory '{}'".format(file_dir))
        os.mkdir(file_dir)
    
    url_filename = url.split('/')[-1]
    if not url_filename and not filename:
        raise Warning("URL does not contain filename, please specify a filename")
    
    filename = url_filename or filename
    
    file_path = pj(file_dir, filename)
    
    # Try cache
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        # Check if file is empty (e.g. failed download from before)
        if not file_size:
            raise Warning("Target file exists but is empty! Please delete file.")
        else:
            print("Hit cache for file with size {:.1f}kB".format(file_size/1E3))
    # Download
    else:
        r = requests.get(url, stream=True)

        with open(file_path, 'wb') as f:
            print("Downloading...")
            # 2**14 = 16kB
            chunks = r.iter_content(chunk_size=2**14)
            # Show progress bar with tqdm
            if show_progress:
                chunks = tqdm(chunks)
            for chunk in chunks:
                f.write(chunk)
    
    # Extract TAR if asked
    if extract_tar:
        # Extract into dir with same name as file, split at first dot
        extract_dirname = pj(file_dir, filename.split('.')[0])
        if not os.path.isdir(extract_dirname):
            print("Extracting TAR archive to directory '{}'".format(extract_dirname))
            tarfile.open(file_path).extractall(extract_dirname)
        else:
            print("Extraction dir exists, giving you its contents '{}'".format(extract_dirname))
        # Return list of absolute paths in extracted dir
        return [pj(extract_dirname, fn) for fn in os.listdir(extract_dirname)]

    return file_path
