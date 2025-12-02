import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

base_folder = Path.cwd()  # path configuration

data_dir = base_folder / "ExportedData"
output_dir = data_dir / "output"
gdb_path = data_dir / "training_samples.gdb" 

CLASS_COLUMN = "class_id"  # id code for each classes

# processing years
years = ["2016", "2020", "2025"]

# PROCESSING LOOP
for year in years:
    print(f"\n" + "="*50)
    print(f"STARTING PIPELINE FOR YEAR: {year}")
    print("="*50)
    
    # Grab the NDVI of each year
    found_images = list(output_dir.glob(f"*{year}*NDVI*.tif"))
    if not found_images:
        print(f"CRITICAL: No NDVI image found for {year}. Skipping...")
        continue
    image_path = found_images[0]
    print(f"Image Source:   {image_path.name}")
    
    # Load the corresponding training data
    layer_name = f"training_{year}"
    print(f"Training Layer: {layer_name}")
    
    try:
        gdf = gpd.read_file(gdb_path, layer=layer_name)
    except Exception as e:
        print(f"ERROR: Could not find layer '{layer_name}' in GDB. Skipping {year}.")
        print(f"Details: {e}")
        continue

    # Flatten 3D polygons if necessary: usually for ArcGISPRO
    from shapely import wkb
    gdf.geometry = gdf.geometry.apply(lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2)))

    # Pixel extraction from the NDVI
    print("Extracting training pixels...")
    training_data = []
    training_labels = []
    
    with rasterio.open(image_path) as src:
        # CRS Check
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
            
        img_data = src.read()
        
        for index, row in gdf.iterrows():
            try:
                # Mask to polygon
                out_image, _ = mask(src, [row.geometry], crop=True)
                
                # Filter out NoData (0)
                valid_mask = np.any(out_image != 0, axis=0)
                pixels = out_image[:, valid_mask].T 
                
                if pixels.size > 0:
                    training_data.append(pixels)
                    labels = np.full((pixels.shape[0],), row[CLASS_COLUMN])
                    training_labels.append(labels)
            except Exception:
                continue
                
    if not training_data:
        print(f"FAILURE: No pixels extracted for {year}. Check polygon overlap.")
        continue

    # Stack data
    X = np.vstack(training_data)
    y = np.concatenate(training_labels)
    print(f"  > Dataset Size: {X.shape[0]} pixels")

    # SPLIT & TRAIN (ACCURACY ASSESSMENT)
    # 70% for training the model, 30% for testing accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # GENERATE REPORT
    print("Evaluating Model Accuracy...")
    y_pred = clf.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred)
    
    print(f"  > Overall Accuracy ({year}): {overall_acc:.2%}")
    print("\nConfusion Matrix & Report:")
    print(classification_report(y_test, y_pred))
    
    # output the result
    report_path = output_dir / f"Accuracy_Report_{year}.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy Report for {year}\n")
        f.write(f"Overall Accuracy: {overall_acc:.2%}\n\n")
        f.write(classification_report(y_test, y_pred))
    print(f"Saved accuracy report to {report_path.name}")

    # 6. full image classification
    print("Classifying full scene...")
    bands, height, width = img_data.shape
    reshaped_img = img_data.reshape(bands, -1).T
    
    # Predict only on valid pixels
    valid_pixels_mask = np.any(reshaped_img != 0, axis=1)
    valid_pixels = reshaped_img[valid_pixels_mask]
    
    prediction_flat = np.zeros(reshaped_img.shape[0], dtype=np.uint8)
    
    if valid_pixels.size > 0:
        prediction_valid = clf.predict(valid_pixels)
        prediction_flat[valid_pixels_mask] = prediction_valid
        
    prediction_map = prediction_flat.reshape(height, width)
    
    # Export Classification
    out_name = output_dir / f"Vienna_{year}_Classification.tif"
    with rasterio.open(image_path) as src:
        profile = src.profile
        profile.update(count=1, dtype=rasterio.uint8, nodata=0)
        
        with rasterio.open(out_name, "w", **profile) as dst:
            dst.write(prediction_map, 1)
            
    print(f"SUCCESS: Saved {out_name.name}")