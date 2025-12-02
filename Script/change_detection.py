
import pandas as pd
# Path Configuration
base_folder = Path.cwd()
data_dir = base_folder / "ExportedData"
output_dir = data_dir / "output"

# Define the comparison you want to make; 2020 was eventually omitted from my final analysis
start_year = "2016"
end_year = "2025"

# Define Class Names
class_names = {
    1: "settlement",
    2: "vegetation",
    3: "waterbody",
    4: "terrain"
}

# Find the files
start_file = list(output_dir.glob(f"*{start_year}_Classification.tif"))[0]
end_file = list(output_dir.glob(f"*{end_year}_Classification.tif"))[0]

print(f"Calculating Change: {start_year} -> {end_year}")

with rasterio.open(start_file) as src_start, rasterio.open(end_file) as src_end:
    # Read data
    start_img = src_start.read(1)
    end_img = src_end.read(1)
    
    # Get spatial metadata for area calculation
    transform = src_start.transform
    # Pixel resolution (usually 10m for Sentinel-2)
    pixel_size_x = transform[0]
    pixel_size_y = -transform[4] # Usually negative
    pixel_area_m2 = pixel_size_x * pixel_size_y
    
    # CHANGE MAP CALCULATION
    # Formula: Start*10 + End
    # Using uint16 because values might go up to 44 or 55
    change_map = (start_img.astype(np.uint16) * 10) + end_img.astype(np.uint16)
    
    # Mask out background (0)
    # If either start or end was 0, the result is invalid
    valid_mask = (start_img != 0) & (end_img != 0)
    change_map = np.where(valid_mask, change_map, 0)

    # Export CHANGE MAP 
    out_name = output_dir / f"Change_Map_{start_year}_{end_year}.tif"
    profile = src_start.profile
    profile.update(dtype=rasterio.uint16, nodata=0)
    
    with rasterio.open(out_name, "w", **profile) as dst:
        dst.write(change_map, 1)
    print(f"Saved Change Map: {out_name.name}")

    # CALCULATE STATISTICS (THE MATRIX) ---
    print("Generating Transition Statistics...")
    
    # Get unique values and counts
    unique, counts = np.unique(change_map[valid_mask], return_counts=True)
    
    transitions = []
    
    for code, count in zip(unique, counts):
        # Decode the math: 21 -> Start=2, End=1
        start_class_id = code // 10
        end_class_id = code % 10
        
        # Calculate Area
        area_m2 = count * pixel_area_m2
        area_km2 = area_m2 / 1_000_000
        area_ha = area_m2 / 10_000
        
        # Get Names
        start_name = class_names.get(start_class_id, f"Class {start_class_id}")
        end_name = class_names.get(end_class_id, f"Class {end_class_id}")
        
        transitions.append({
            "Transition_Code": code,
            "From_Class": start_name,
            "To_Class": end_name,
            "Pixel_Count": count,
            "Area_km2": round(area_km2, 4),
            "Area_Hectares": round(area_ha, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(transitions)
    
    # Sort by Area to see biggest changes first
    df = df.sort_values(by="Area_km2", ascending=False)
    
    # EXPORT CSV OF THE CHANGE DETAILS
    csv_name = output_dir / f"Change_Stats_{start_year}_{end_year}.csv"
    df.to_csv(csv_name, index=False)
    
    print("\n TOP 5 MAJOR CHANGES")
    print(df.head(5))
    print(f"\nFull statistics saved to: {csv_name.name}")