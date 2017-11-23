# Path handling and other niceties for easy reference and less headache
import os
import tarfile
import requests

pj = os.path.join

# abspath also norms the path
data_dir = os.path.abspath(pj(os.path.dirname(__file__), '..', 'data'))

# Path config based on file category
RADAR_REFL_COMP_DIR = pj(data_dir, 'radar_refl_comp')


def download_cache_ftp(file_dir, ftp_conn, ftp_url, filename=None, verbose=False):
    """
    Retrieve binary file from FTP server if it doesn't exist yet.
    """
    # Create parent dir if needed
    if not os.path.exists(file_dir):
        if verbose:
            print("Creating directory '{}'".format(file_dir))
        os.makedirs(file_dir)
    
    url_filename = ftp_url.split('/')[-1]
    if not url_filename and not filename:
        raise Warning("URL does not contain filename, please specify a filename")
    
    filename = url_filename or filename
    
    file_path = pj(file_dir, filename)
    
    # Try cache
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        # Check if file is empty (e.g. failed download from before)
        if not file_size:
            raise Warning("Target file exists but is empty! Please delete file. {}".format(file_path))
        else:
            if verbose:
                print("Hit cache for file with size {:.1f}kB".format(file_size/1E3))
    # Download
    else:
        with open(file_path, 'wb') as f:
            if verbose:
                print("Downloading...")

            def callback(chunk):
                f.write(chunk)
            # Blocks until complete
            ftp_conn.retrbinary('RETR {}'.format(ftp_url), callback, blocksize=102400, rest=None)
            if verbose:
                print("Download finished")
    
#     # Extract TAR if asked
#     if extract_tar:
#         # Extract into dir with same name as file, split at first dot
#         extract_dirname = pj(file_dir, filename.split('.')[0])
#         if not os.path.isdir(extract_dirname):
#             print("Extracting TAR archive to directory '{}'".format(extract_dirname))
#             tarfile.open(file_path).extractall(extract_dirname)
#         else:
#             print("Extraction dir exists, giving you its contents '{}'".format(extract_dirname))
#         # Return list of absolute paths in extracted dir
#         return [pj(extract_dirname, fn) for fn in os.listdir(extract_dirname)]

    return file_path
