import gzip

gzipped_file = 'Depression_detection_using_Twitter_post-master\GoogleNews-vectors-negative300.bin.gz'

with gzip.open(gzipped_file, 'rb') as f:
    bin_data = f.read()

# Print the binary data as a hexadecimal string or decode it as needed
print(bin_data.hex())