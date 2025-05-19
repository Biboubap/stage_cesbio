from osgeo import gdal
import numpy as np

def load_classification_and_rgb(classif_tif, rgb_tif):
    ds_classif = gdal.Open(classif_tif)
    classif = ds_classif.GetRasterBand(1).ReadAsArray()
    ds_rgb = gdal.Open(rgb_tif)
    rgb = np.stack([ds_rgb.GetRasterBand(i+1).ReadAsArray() for i in range(3)], axis=-1)
    return classif, rgb, ds_classif, ds_rgb

def create_lichen_mask(classif, nodata_val=255):
    mask = (classif == 1).astype(np.uint8)
    mask[classif == 0] = nodata_val  # NoData pour QGIS
    return mask

def apply_mask_to_rgb(rgb, mask):
    lichen_rgb = np.zeros_like(rgb)
    nonlichen_rgb = np.zeros_like(rgb)
    for c in range(3):
        lichen_rgb[..., c] = np.where(mask == 1, rgb[..., c], 0)
        nonlichen_rgb[..., c] = np.where((mask != 1) & (mask != 255), rgb[..., c], 0)
    return lichen_rgb, nonlichen_rgb

def save_mask(mask, ref_ds, out_mask_tif, nodata_val=255):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_mask_tif, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(mask)
    out_ds.GetRasterBand(1).SetNoDataValue(nodata_val)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None

def save_rgb(rgb_img, ref_ds, out_tif):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif, rgb_img.shape[1], rgb_img.shape[0], 3, gdal.GDT_Byte)
    for c in range(3):
        out_ds.GetRasterBand(c+1).WriteArray(rgb_img[..., c])
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None

def process_lichen_extraction(classif_tif, rgb_tif, out_mask_tif, out_lichen_tif, out_nonlichen_tif):
    classif, rgb, ds_classif, ds_rgb = load_classification_and_rgb(classif_tif, rgb_tif)
    mask = create_lichen_mask(classif)
    lichen_rgb, nonlichen_rgb = apply_mask_to_rgb(rgb, mask)
    save_mask(mask, ds_classif, out_mask_tif)
    save_rgb(lichen_rgb, ds_rgb, out_lichen_tif)
    save_rgb(nonlichen_rgb, ds_rgb, out_nonlichen_tif)
    print("Fichiers sauvegardés :")
    print(out_mask_tif)
    print(out_lichen_tif)
    print(out_nonlichen_tif)

# Exemple d'utilisation
if __name__ == "__main__":
    process_lichen_extraction(
        classif_tif="data/samples/selection4/classification_result_3.tif",
        rgb_tif="data/rgb_reshaped.tif",
        out_mask_tif="data/samples/selection4/lichen_mask.tif",
        out_lichen_tif="data/samples/selection4/lichen_rgb.tif",
        out_nonlichen_tif="data/samples/selection4/nonlichen_rgb.tif"
    )