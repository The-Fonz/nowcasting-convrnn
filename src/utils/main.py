# Path handling and other niceties for easy reference and less headache
import os, io, re, glob, tarfile, itertools
import concurrent.futures

from tqdm import tqdm_notebook

pj = os.path.join

# abspath also norms the path
data_dir = os.path.abspath(pj(os.path.dirname(__file__), '..', '..', 'data'))

# Path config based on file category
RADAR_REFL_COMP_DIR = pj(data_dir, 'radar_refl_comp')
RADAR_FORECAST_REFL_COMP_DIR = pj(data_dir, 'radar_forecast_refl_comp')


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
    
    filename = filename or url_filename

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
        try:
            with open(file_path, 'wb') as f:
                if verbose:
                    print("Downloading to {}".format(file_path))

                def callback(chunk):
                    f.write(chunk)
                # Blocks until complete
                ftp_conn.retrbinary('RETR {}'.format(ftp_url), callback, blocksize=102400, rest=None)
        
        except KeyboardInterrupt as e:
            print("Received KeyboardInterrupt, deleting file to avoid incomplete files")
            os.remove(file_path)
            raise e

        if verbose:
            print("Download finished")

    return file_path


def explain_hdf5_file(file):
    """
    Show groups, datasets and attributes in h5py file
    :param file: h5py file
    :returns: string
    """
    out = []
    file.visititems(lambda name, obj:
         out.append('{}\n\t{}\n\t\t{}\n'
        .format(
             name,
             obj,
            '\n\t\t'.join('{0}: {1}'.format(*t) for t in obj.attrs.items()))
        ))
    return ''.join(out)


def untar_concurrent(tar_files, delete=True, max_workers=8):
    """
    Untar all files given.
    :param filelist: List of files to untar
    :kwarg delete: Delete source file after verifying successful untar
    """
    def untar_file(tfp):
        t = tarfile.open(tfp)
        folder = os.path.dirname(tfp)
        try:
            contents_list = t.getnames()
            t.extractall(folder)
        except tarfile.ReadError as e:
            print("Failed to read file {}".format(tfp))
            raise e
        # Check if files exist
        for f in contents_list:
            path = os.path.join(folder, f)
            if not os.path.isfile(os.path.join(path)):
                raise Warning("%s not found!", path)
        # Remove original file to avoid doubling disk space
        if delete:
            os.remove(tfp)

    # Untar concurrently for massive speedup
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as e:
        tasks = [e.submit(untar_file, tfp) for tfp in tar_files]
        print("Submitted all tasks")
        # Wait for finish and show progress bar
        for future in tqdm_notebook(concurrent.futures.as_completed(tasks), desc="Extracting", unit="file"):
            # Retrieve exceptions
            # NOTE: These are raised after all futures have completed!
            # tqdm progress bar will stop working 
            future.result()


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks. grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def map_reduce_concurrent(arg_list, map_func, reduce_func, n_workers=8, chunk_size=1000):
    """
    Concurrently apply accumulating map function for large datasets where map results don't fit in memory.
    
    :param arg_list: List of arguments for reduce_func
    :param map_func: Gets a chunk of arg_list to process, returns accumulated value
    :param reduce_func: Gets list of accumulated values from all map funcs, and length of arg_list
    :returns: Output of reduce_func
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as tpe:
        # Divide arg_list into chunks and delete None's
        chunks = [[v for v in chunk if v is not None] for chunk in grouper(arg_list, chunk_size)]
        futs = [tpe.submit(map_func, chunk) for chunk in chunks]
        results = []
        with tqdm_notebook(total=len(futs)*chunk_size) as pbar:
            for fut in concurrent.futures.as_completed(futs):
                pbar.update(chunk_size)
                results.append(fut.result())
    return reduce_func(results, len(arg_list))
