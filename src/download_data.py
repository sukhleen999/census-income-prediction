'''
This script aims to download dataset from online source. 
This script takes the dataset link and output directory name.

Usage: download_data.py <url> --out_dir=<out_dir> [--file_name=<file_name>]

Options:
<url>                     URL to the dataset
--out_dir=<out_dir>       Path to the data folder
[--file_name=<file_name>] Output file name
'''
from docopt import docopt
import os
import urllib.request

opt = docopt(__doc__)

def main():
    # Retrieve text data from URL
    with urllib.request.urlopen(opt['<url>']) as fp:
        content = fp.read().decode('utf-8')

    # Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])

    # Set output filename, default use filename of the remote file
    filename = opt['--file_name']
    if filename is None:
        filename = os.path.split(opt['<url>'])[-1]

    # Write content to the output file
    with open(os.path.join(opt['--out_dir'], filename), 'w') as fp:
        fp.write(content)
        
    print("done")    

if __name__ == "__main__":
    main()