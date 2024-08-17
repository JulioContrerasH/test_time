import os
import mlstac
import rasterio as rio
import time
import pandas as pd

metadata = mlstac.load_metadata(r'D:/GitHub/mlstac_dataloader/data/test/test_2000_high.mlstac')
metadata = metadata.iloc[0:5] # sample beacause weighs a lot

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


data_types = {
    "byte": "uint8",
    "int16": "int16",
    "float32": "float32"
}

results = []

for i, row in metadata.iterrows():
    dataset, dataset_metadata = mlstac.load_data(metadata[i:i+1], save_metadata_datapoint=True)[0]
    dataset_metadata['driver'] = 'GTiff'
    datapoint_id = metadata['datapoint_id'].iloc[i]
    for dtype_name, dtype_value in data_types.items():
        dataset_metadata['dtype'] = dtype_value
        for name, options in compression_options.items():
            output_path = f"D:/GitHub/mlstac_dataloader/data/test/test_2000_high/{name}/{dtype_name}/{datapoint_id}.tif"
            if os.path.exists(output_path):
                print(f"Archivo {output_path} ya existe. Saltando...")
                continue

            if 'PREDICTOR' in options and options['PREDICTOR'] == 3:
                if dataset_metadata['dtype'] not in ['float32', 'float64']:
                    print(f"Saltando {name} para {datapoint_id} debido a tipo de dato incompatible ({dataset_metadata['dtype']})")
                    continue

            compression_metadata = {**dataset_metadata, **options}
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Guardando {name} para {datapoint_id}.")
            start_time = time.time()
            with rio.open(output_path, 'w', **compression_metadata) as dst:
                dst.write(dataset)
            end_time = time.time()
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # tamaño del archivo en MB
            elapsed_time = end_time - start_time

            if elapsed_time == 0 or file_size == 0:
                print(f"Omitiendo {name} para {datapoint_id} debido a tiempo transcurrido nulo o tamaño de archivo nulo.")
                continue

            write_speed = file_size / elapsed_time

            results.append({
                "compression": name,
                "data_type": dtype_name,
                "datapoint_id": datapoint_id,
                "write_speed": write_speed
            })

df_results = pd.DataFrame(results)

summary = df_results.groupby(['compression', 'data_type']).agg(
    average_speed=('write_speed', 'mean'),
    std_dev_speed=('write_speed', 'std')
).reset_index()

summary.columns = [
    "Compression Method",               
    "Data Type",                        
    "Average Write Speed (MB/s)",       
    "Standard Deviation of Write Speed (MB/s)" 
]

summary_sorted = summary.sort_values(by="Average Write Speed (MB/s)", ascending=True).reset_index(drop=True)

summary_sorted.to_csv("D:/GitHub/mlstac_dataloader/data/test/test_2000_high/Write_speed_summary_2000.csv", index=False)
