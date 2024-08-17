import os
import time
import pandas as pd
import rasterio as rio

# Define the base directory paths
base_dir = "D:/GitHub/mlstac_dataloader/data/test/test_2000_high"
uncompressed_dir = f"{base_dir}/none"

# Define the compression options and data types used
compression_options = {
    "none": {},  # Sin compresión
    "packbits": {"COMPRESS": "PACKBITS"},
    "deflate": {"COMPRESS": "DEFLATE"},
    "deflate_pred2": {"COMPRESS": "DEFLATE", "PREDICTOR": 2},
    "deflate_pred3": {"COMPRESS": "DEFLATE", "PREDICTOR": 3},
    "deflate_zlev9": {"COMPRESS": "DEFLATE", "ZLEVEL": 9},
    "deflate_zlev1": {"COMPRESS": "DEFLATE", "ZLEVEL": 1},
    "deflate_zlev1_pred2": {"COMPRESS": "DEFLATE", "ZLEVEL": 1, "PREDICTOR": 2},
    "deflate_zlev1_pred3": {"COMPRESS": "DEFLATE", "ZLEVEL": 1, "PREDICTOR": 3},
    "lzma": {"COMPRESS": "LZMA"},
    "lzw": {"COMPRESS": "LZW"},
    "lzw_pred2": {"COMPRESS": "LZW", "PREDICTOR": 2},
    "lzw_pred3": {"COMPRESS": "LZW", "PREDICTOR": 3},
    "zstd": {"COMPRESS": "ZSTD"},
    "zstd_pred2": {"COMPRESS": "ZSTD", "PREDICTOR": 2},
    "zstd_pred3": {"COMPRESS": "ZSTD", "PREDICTOR": 3},
    "zstd_zlev15": {"COMPRESS": "ZSTD", "ZSTD_LEVEL": 15},
    "zstd_zlev1": {"COMPRESS": "ZSTD", "ZSTD_LEVEL": 1},
    "zstd_zlev1_pred2": {"COMPRESS": "ZSTD", "ZSTD_LEVEL": 1, "PREDICTOR": 2},
    "zstd_zlev1_pred3": {"COMPRESS": "ZSTD", "ZSTD_LEVEL": 1, "PREDICTOR": 3},
}
compression_options = list(compression_options.keys())
compression_options = compression_options[1:]

data_types = ["byte", "int16", "float32"]

# Initialize a DataFrame to store the results
results = []

# List all uncompressed files
uncompressed_files = {}
for dtype_name in data_types:
    dtype_dir = os.path.join(uncompressed_dir, dtype_name)
    if os.path.exists(dtype_dir):
        files = os.listdir(dtype_dir)
        uncompressed_files[dtype_name] = [os.path.join(dtype_dir, f) for f in files]

# Process each file
for dtype_name, file_paths in uncompressed_files.items():
    for uncompressed_file_path in file_paths:

        # Get the base filename (datapoint_id) from the uncompressed file
        base_filename = os.path.basename(uncompressed_file_path)
        
        # Measure the read time of the uncompressed file
        start_time = time.time()
        with rio.open(uncompressed_file_path, 'r') as src:
            uncompressed_data = src.read()
        end_time = time.time()
        
        uncompressed_read_time = end_time - start_time
        uncompressed_size = os.path.getsize(uncompressed_file_path) / (1024 * 1024)  # in MB
        
        if uncompressed_read_time == 0:
                uncompressed_speed = 0  # Velocidad en MB/s4
        else:
            uncompressed_speed = uncompressed_size / uncompressed_read_time  # Velocidad en MB/s

        # Now compare against each compressed file
        for name in compression_options:
            compressed_file_path = os.path.join(base_dir, name, dtype_name, base_filename)
            
            if not os.path.exists(compressed_file_path):
                print(f"Archivo comprimido no encontrado: {compressed_file_path}")
                continue

            # Measure the read time of the compressed file
            start_time = time.time()
            with rio.open(compressed_file_path, 'r') as src:
                compressed_data = src.read()
            end_time = time.time()

            compressed_read_time = end_time - start_time
            compressed_size = os.path.getsize(compressed_file_path) / (1024 * 1024)  # in MB

            if compressed_read_time == 0:
                compressed_speed = 0  # Velocidad en MB/s4
            else:
                compressed_speed = compressed_size / compressed_read_time
            # Calculate the compression rate
            compression_rate = uncompressed_size / compressed_size
            
            # Store the results
            results.append({
                "compression": name,
                "data_type": dtype_name,
                "datapoint_id": base_filename,
                "uncompressed_read_time": uncompressed_read_time,
                "uncompressed_speed_MB_s": uncompressed_speed,
                "compressed_read_time": compressed_read_time,
                "compressed_speed_MB_s": compressed_speed,
                "compression_rate": compression_rate
            })
            print(results)
            
# Convert the results into a DataFrame
df_results = pd.DataFrame(results)
df_results = df_results[df_results.compressed_speed_MB_s != 0]
df_results = df_results[df_results.uncompressed_speed_MB_s != 0]

# Calculate average and standard deviation for each metric
summary = df_results.groupby(['compression', 'data_type']).agg(
    average_uncompressed_read_time=('uncompressed_read_time', 'mean'),
    std_dev_uncompressed_read_time=('uncompressed_read_time', 'std'),
    average_uncompressed_speed_MB_s=('uncompressed_speed_MB_s', 'mean'),
    std_dev_uncompressed_speed_MB_s=('uncompressed_speed_MB_s', 'std'),
    average_compressed_read_time=('compressed_read_time', 'mean'),
    std_dev_compressed_read_time=('compressed_read_time', 'std'),
    average_compressed_speed_MB_s=('compressed_speed_MB_s', 'mean'),
    std_dev_compressed_speed_MB_s=('compressed_speed_MB_s', 'std'),
    average_compression_rate=('compression_rate', 'mean'),
    std_dev_compression_rate=('compression_rate', 'std'),
).reset_index()


summary_sorted = summary.sort_values(by="average_compressed_speed_MB_s", ascending=True).reset_index(drop=True)

summary_sorted.columns = [
    "Compression Method",                   
    "Data Type",                            
    "Avg. Uncompressed Read Time (s)",      
    "Std Dev of Uncompressed Read Time (s)",
    "Avg. Uncompressed Speed (MB/s)",       
    "Std Dev of Uncompressed Speed (MB/s)", 
    "Avg. Compressed Read Time (s)",        
    "Std Dev of Compressed Read Time (s)",  
    "Avg. Compressed Speed (MB/s)",         
    "Std Dev of Compressed Speed (MB/s)",   
    "Avg. Compression Rate",                
    "Std Dev of Compression Rate"           
]
summary_sorted.to_csv("D:/GitHub/mlstac_dataloader/data/test/test_2000_high/Read_speed_rate_summary_2000.csv", index=False)
