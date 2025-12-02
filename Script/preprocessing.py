from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
from glob import glob

# creating an absolute path to access the project's folder
base = Path.cwd()

# Fixed path joining (removed redundant 'vienna_timeseries' logic if base is already the root)
data_dir = base / "ExportedData"

years = {
    "2016": data_dir / "s2_2016",
    "2020": data_dir / "s2_2020",
    "2025": data_dir / "s2_2025"
}

roi_shp = data_dir / "roi" / "ROI.shp"  # variable for boundary shapefile
output_dir = data_dir / "output"  #variable for outputs
output_dir.mkdir(exist_ok=True, parents=True) # parents=True just in case intermediate folders are missing

# Load boundary shapefile
roi = gpd.read_file(roi_shp)

# set cordinate transformation incase the roi projection doesn't match imageries projection.
target_crs = "EPSG:32633"
if roi.crs !=target_crs:
    print(f" Reprojecting ROI from {roi_crs} to {target_crs}")
    roi = roi.to_crs(target_crs) 

# Create the masking geometry
roi_geom = [roi.geometry.unary_union]

print("Setup complete. ROI CRS:", roi.crs)


# Defining the bands being stacked (10m resolution only)
REQUIRED_BANDS = ['B02', 'B03', 'B04', 'B08']

for year, folder_path in years.items():
    print(f"\n Processing Year: {year}")

    masked_layers = []
    output_transform = None

    try: 
        # loop through Blue, Green, Red and NIR Bands
        for band_name in REQUIRED_BANDS:
            # find all the bands in all scenes.
            all_files = list(folder_path.rglob(f"*{band_name}*.jp2"))

            # filtering for the bands in IMG_DATA and also just 10m resolution to avoid stacking 20m and 60m bands too
            band_files = []
            for f in all_files:
                if "IMG_DATA" not in str(f):
                    continue
                if "10m" in f.name or "R10m" in str(f.parent):
                    band_files.append(f)
                    # a block incase L1C band without 10m res is in dataset
            if not band_files and all_files:
                print(f" Note: '10m' label not found (likely L1C data). Using found files")
                band_files = [f for f in all_files if "IMG_DATA" in str(f)]
                # one last check for right band
            if not band_files:
                print (f" WARNING: No files found for {band_name} in {years}")
                continue
            print(f" {band_name}) : Found {len(band_files)} tiles. Merging...")

            # FILE OPENING, MERGING , FILE CLOSING
            src_files_to_mosaic = [rasterio.open(f) for f in band_files]
            mosaic, out_trans = merge(src_files_to_mosaic)
            for src in src_files_to_mosaic:
                src.close()

            # PREPARING METADATA FOR MEMORYFILE
            temp_meta = src_files_to_mosaic[0].meta.copy()
            temp_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1,
            })

            # CROPPING(MASKING USING MemoryFile)
            with MemoryFile() as memfile:
                with memfile.open(**temp_meta) as dataset:
                    dataset.write(mosaic)
                    crop_image, crop_transform = mask(dataset, roi_geom, crop=True)
                    masked_layers.append(crop_image[0])

                    if output_transform is None:
                        output_transform = crop_transform

        # BAND STACKING AND SAVING
        if len(masked_layers) == 4:
            final_stack = np.array(masked_layers)
            final_meta = temp_meta.copy()
            final_meta.update({
                "driver": "GTiff",
                "height": final_stack.shape[1],
                "width": final_stack.shape[2],
                "transform": output_transform,
                "count": 4,
                "dtype": final_stack.dtype
            })
            out_name = output_dir / f" Vienna_{year}_merged_masked.tif"
            with rasterio.open(out_name, "w", **final_meta) as dest:
                dest.write(final_stack)
                dest.set_band_description(1, "Blue")
                dest.set_band_description(2, "Green")
                dest.set_band_description(3, "Red")
                dest.set_band_description(4, "NIR")

            print(f" SUCCESS: Saved {out_name.name} (Covering Full ROI)")
        else:
            print(f" Error: Could not complete{year}. Missing ands.")

    except Exception as e:
        print(f" Error processing {year}: {e}")

### You can run this in another script.
# INDICES CALCULATION(NDVI). This was done to extract the radiance from the images after training the samples for the model
base = Path.cwd()
output_dir = base.parent / "ExportedData" / "output" 

# Verify paths
tif_files = list(output_dir.glob("*_merged_masked.tif"))
if not tif_files:
    # Try looking in current folder if the above failed; safety measure
    output_dir = base / "ExportedData" / "output"
    tif_files = list(output_dir.glob("*_merged_masked.tif"))

print(f"Found {len(tif_files)} images to process in: {output_dir}")

def calculate_ndvi(nir, red):
    """Calculates NDVI and handles division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
    # Fill NaN/Inf with 0 (NoData)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    return ndvi

for fp in tif_files:
    print(f"\nProcessing: {fp.name}")
    
    with rasterio.open(fp) as src:
        profile = src.profile.copy()
        
        # Read bands (Blue=1, Green=2, Red=3, NIR=4)
        red = src.read(3).astype(float)
        nir = src.read(4).astype(float)
        
        # Read the whole stack to append to it
        all_bands = src.read()
        
        # Calculate NDVI
        ndvi = calculate_ndvi(nir, red)
        
        # Expand dimension to make it (1, H, W) for stacking
        ndvi_expanded = np.expand_dims(ndvi, axis=0)
        
        # Create 5-band stack
        final_stack = np.concatenate((all_bands, ndvi_expanded), axis=0)
        
        #  Updating METADATA
        profile.update(
            count=5,
            dtype=rasterio.float32,
            nodata=0  # instruction for software to classify 0 as transparent/nothing
        )
        
    out_name = output_dir / f"{fp.stem}_NDVI.tif"
    
    with rasterio.open(out_name, "w", **profile) as dst:
        dst.write(final_stack.astype(rasterio.float32))
        
        # Label bands for ArcGISPRO
        dst.set_band_description(1, 'Blue')
        dst.set_band_description(2, 'Green')
        dst.set_band_description(3, 'Red')
        dst.set_band_description(4, 'NIR')
        dst.set_band_description(5, 'NDVI')
        
    print(f"SUCCESS: Saved {out_name.name}")

    ### congratulations, pre-processing done