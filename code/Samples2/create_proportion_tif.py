"""
Create a multi-band TIFF file containing the proportions of each class on Sentinel-2 samples.
Each band represents a different class proportion, scaled from 0-100 (percentage).
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import argparse
from tqdm import tqdm

def create_proportion_tiff(input_csv, sentinel_path, output_path):
    """
    Create a multi-band TIFF from class proportion data from a CSV file.
    
    Args:
        input_csv: Path to CSV containing class proportions (from sentinel_proportion.py)
        sentinel_path: Path to a Sentinel-2 raster for georeference
        output_path: Path to save the output multi-band TIFF
    """
    print(f"Creating proportion TIFF from {input_csv}")
    
    # Load class proportions data
    df = pd.read_csv(input_csv)
    
    # Define classes and get max row/col coordinates
    classes = [
        "chicoutai",
        "dry_depression",
        "green_depression", 
        "lichen",
        "sphaignes",
        "watered_depression",
        "black_depression",
        "none"
    ]
    
    # Calculate derived classes
    df['chicoutai_green'] = df['chicoutai'] + df['green_depression']
    df['through_proportion'] = df['sphaignes'] + df['dry_depression'] + df['black_depression'] + df['watered_depression']
    
    all_classes = classes + ['chicoutai_green', 'through_proportion']
    
    # Get dimensions of Sentinel raster
    ds_sentinel = gdal.Open(sentinel_path)
    if ds_sentinel is None:
        raise ValueError(f"Could not open Sentinel raster: {sentinel_path}")
        
    sentinel_width = ds_sentinel.RasterXSize
    sentinel_height = ds_sentinel.RasterYSize
    geo_transform = ds_sentinel.GetGeoTransform()
    projection = ds_sentinel.GetProjection()
    
    # Create output raster (multi-band)
    driver = gdal.GetDriverByName('GTiff')
    num_bands = len(all_classes)
    out_ds = driver.Create(output_path, sentinel_width, sentinel_height, num_bands, gdal.GDT_Int16, 
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    if out_ds is None:
        raise ValueError(f"Could not create output file {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Initialize arrays with zeros for each band
    proportion_arrays = {class_name: np.zeros((sentinel_height, sentinel_width), dtype=np.int16) 
                        for class_name in all_classes}
    
    # Fill arrays with proportion values (scaled to 0-100)
    print("Filling proportion arrays...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        col_s = int(row['col_s'])
        row_s = int(row['row_s'])
        
        # Skip if out of bounds
        if row_s >= sentinel_height or col_s >= sentinel_width:
            continue
            
        # Set proportion values for each class (scaled to 0-100)
        for class_name in all_classes:
            if class_name in row:
                # Multiply by 100 to get percentage (0-100) and convert to int16
                proportion_arrays[class_name][row_s, col_s] = int(row[class_name] * 100)
    
    # Write arrays to bands
    print("Writing bands to output TIFF...")
    for i, class_name in enumerate(all_classes, start=1):
        band = out_ds.GetRasterBand(i)
        band.WriteArray(proportion_arrays[class_name])
        band.SetDescription(class_name)
        band.SetNoDataValue(-1)  # Use -1 as NoData value
        band.FlushCache()
    
    # Add band descriptions as metadata
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": class_name for i, class_name in enumerate(all_classes)})
    
    # Close dataset
    out_ds = None
    print(f"Multi-band proportion TIFF created at {output_path}")
    
    # Create a color table for visualization in QGIS
    create_color_interpretation_file(output_path, all_classes)

def create_color_interpretation_file(tiff_path, classes):
    """
    Create a color interpretation file for QGIS to properly display the bands
    
    Args:
        tiff_path: Path to the multi-band TIFF file
        classes: List of class names in band order
    """
    # Define a consistent color scheme for classes
    colors = {
        "chicoutai": "#006400",        # Dark green
        "dry_depression": "#998800",    # Yellow-brown
        "green_depression": "#32CD32",  # Lime green
        "lichen": "#C8C8C8",           # Light gray
        "sphaignes": "#8B4513",         # Saddle brown
        "watered_depression": "#505050", # Dark gray
        "black_depression": "#321400",   # Very dark brown
        "none": "#000000",              # Black
        "chicoutai_green": "#228B22",   # Forest green
        "through_proportion": "#A0522D" # Sienna
    }
    
    # Create .vrt file with color interpretation
    vrt_path = tiff_path.replace('.tif', '.vrt')
    
    ds = gdal.Open(tiff_path)
    vrt_options = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(vrt_path, [tiff_path], options=vrt_options)
    
    # Create metadata file with band descriptions
    # This helps QGIS show proper names in the layer properties
    qml_path = tiff_path.replace('.tif', '.qml')
    with open(qml_path, 'w') as f:
        f.write("""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.4-Białowieża">
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" name="name" value=""/>
      <Option name="properties"/>
      <Option type="QString" name="type" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" type="singlebandpseudocolor" band="1" classificationMin="0" classificationMax="100">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="100" classificationMode="1" colorRampType="INTERPOLATED" clip="0" labelPrecision="0" minimumValue="0">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option type="QString" name="color1" value="#ffffff"/>
              <Option type="QString" name="color2" value="{}"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
            </Option>
          </colorramp>
          <item label="0%" alpha="0" color="#ffffff" value="0"/>
          <item label="50%" alpha="128" color="{}" value="50"/>
          <item label="100%" alpha="255" color="{}" value="100"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" invertColors="0" colorizeBlue="128" grayscaleMode="0" colorizeOn="0" saturation="0" colorizeRed="255" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
""".format(colors.get(classes[0], "#ff0000"), 
            colors.get(classes[0], "#ff0000"), 
            colors.get(classes[0], "#ff0000")))
    
    print(f"Created color interpretation files: {vrt_path} and {qml_path}")
    print("NOTE: In QGIS, use the 'Select Band' option in layer properties to view each class proportion")

if __name__ == "__main__":
    wap = 32
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    use_peat = False
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "" if superresolution else "_10m"
    
    # Set path modifiers based on superresolution flag
    mediane_dir = "mediane" if superresolution else "mediane_10m"
    file_prefix = "" if superresolution else "10m_"
    
    input_csv = f"data/samples/selection14/regression_wap{wap}_no_chicoutai{peat_suffix}{resolution_suffix}/class_proportions_WAP{wap}.csv"
    sentinel_path = f"DataCubeS2/BandsS22023_WAP{wap}{peat_suffix}/{mediane_dir}/{file_prefix}mediane_clipped_STACK_2023_BandB2_WAP{wap}_deflate.tif"
    output_path = f"data/samples/selection14/regression_wap{wap}_no_chicoutai{peat_suffix}{resolution_suffix}/proportions_WAP{wap}.tif"
    create_proportion_tiff(input_csv, sentinel_path, output_path)

