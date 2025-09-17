import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
import os
mod_a0 = r"D:\methane\mod\clean\run_bg_A000_W1416.csv"
mod_a10 = r"D:\methane\mod\clean\run_bg_A100_W1416.csv"
hisui = r"D:\methane\spectral_data_sample_HSHL1G_N345E1354_20201027020705_20221013062213_x840_y1175.csv"
band = r"D:\methane\HSHL1G_N402E0583_20210803092156_20240106123056_B.csv"

bandinfo = pd.read_csv(band)
hisui_spec = pd.read_csv(hisui)
mod_a0 = pd.read_csv(mod_a0)
mod_a10 = pd.read_csv(mod_a10)
print(hisui_spec)

bandinfo_cols = bandinfo.columns.tolist()
hisui_cols = hisui_spec.columns.tolist()
mod_cols = mod_a0.columns.tolist()
print(bandinfo_cols, hisui_cols, mod_cols)

mod_a0_conv = mod_a0.copy()
mod_a10_conv = mod_a10.copy()
mod_a0_conv["radiance_wm2_sru_um"] = mod_a0_conv["radiance"] * 10
mod_a10_conv["radiance_wm2_sru_um"] = mod_a10_conv["radiance"] * 10 #Unit conversion
print(mod_a0_conv)
print(mod_a10_conv)

target_anchor_nm = 2139.0 #Using 2139 nm for albedo estimation
bandinfo["abs_diff_anchor"] = (bandinfo["CenterWavelengthNanometer"] - target_anchor_nm).abs() #Difference from 2139 (absolute value)
anchor_row = bandinfo.loc[bandinfo["abs_diff_anchor"].idxmin()] #Extract the row with the smallest difference
anchor_bandno = int(anchor_row["BandNo"])
anchor_center = float(anchor_row["CenterWavelengthNanometer"])
anchor_fwhm = float(anchor_row["FullWidthAtHalfMaximumNanometer"])
print(anchor_bandno, anchor_center, anchor_fwhm)

ch4_min, ch4_max = 2248.0, 2298.0 #From the paper (since it became 6 items over there, should we adjust it to 6 items?)
ch4_bands = bandinfo[(bandinfo["CenterWavelengthNanometer"] >= ch4_min) & 
                     (bandinfo["CenterWavelengthNanometer"] <= ch4_max)].copy()
ch4_bands = ch4_bands.sort_values("CenterWavelengthNanometer")
ch4_bands[["BandNo", "CenterWavelengthNanometer", "FullWidthAtHalfMaximumNanometer"]],len(ch4_bands)

def gaussian_weights(wave_nm, center_nm, fwhm_nm):
    sigma = fwhm_nm / (2 * math.sqrt(2 * math.log(2)))  #FWHM -> sigma
    w = np.exp(-0.5 * ((wave_nm - center_nm) / sigma) ** 2)
    w_sum = w.sum()
    if w_sum == 0:
        return w
    return w / w_sum

def convolve_band(wave_nm, rad, center_nm, fwhm_nm): #High-resolution spectral rad(λ) weighted by the RSR weight w(λ) for that band → the radiant brightness actually observed by that band
    w = gaussian_weights(wave_nm, center_nm, fwhm_nm)
    return float(np.sum(rad * w))

wave_nm = mod_a0_conv["wave_nm"].values
rad_a0 = mod_a0_conv["radiance_wm2_sru_um"].values
rad_a10 = mod_a10_conv["radiance_wm2_sru_um"].values

L_path = convolve_band(wave_nm, rad_a0, anchor_center, anchor_fwhm) #Brightness observed by the sensor at 0% reflectance
L10_anchor = convolve_band(wave_nm, rad_a10, anchor_center, anchor_fwhm)
print(wave_nm, rad_a0, rad_a10, L_path, L10_anchor)

hisui_waves = hisui_spec["Wavelength"].values
hisui_rads = hisui_spec["Radiance"].values
idx_anchor_obs = int(np.argmin(np.abs(hisui_waves - anchor_center))) #ac=2137.525nm, each wavelength and its closest idx
lambda_anchor_obs = float(hisui_waves[idx_anchor_obs]) #The observed wavelength closest to the standard
L_av_anchor = float(hisui_rads[idx_anchor_obs]) #Radiative brightness at this time
print(idx_anchor_obs,lambda_anchor_obs,L_av_anchor)

A = (L_av_anchor - L_path) / (L10_anchor - L_path) * 0.10 #Equation (1) in the paper
print(A)

rows = []
chi = 0.10 / A #χ = A(LUTb)/A(estimated) 
for i, r in ch4_bands.iterrows():
    bandno = int(r["BandNo"])
    center = float(r["CenterWavelengthNanometer"])
    fwhm = float(r["FullWidthAtHalfMaximumNanometer"])
    Lb = convolve_band(wave_nm, rad_a10, center, fwhm)  #Observed brightness at 10%
    idx_obs = int(np.argmin(np.abs(hisui_waves - center)))
    L_av = float(hisui_rads[idx_obs])
    lambda_obs = float(hisui_waves[idx_obs])
    
    L_res = Lb - chi * L_av
    
    rows.append({
        "BandNo": bandno,
        "Center_nm": center,
        "FWHM_nm": fwhm,
        "L_b": Lb,
        "L_av": L_av,
        "chi": chi,
        "L_res": L_res
    })
res_df = pd.DataFrame(rows)
print(res_df)

N = len(res_df)
sum_res = float(res_df["Residual(L_b - chi*L_av)"].sum())
C_value = (sum_res / (N * A)) * 100 #Unlike the paper, I was currently calculating in um, so the scaling factor to convert to nm units is 100
print(C_value)

# read radiometric file (csv)
def read_bfile(file):
    #Create a new filename by adding “_B.csv” to the base filename
    fileb = os.path.splitext(file)[0] + "_B.csv"
    #Reading CSV files using pandas 
    df = pd.read_csv(fileb)
    #Initialize the parameter array 
    param = np.zeros((185, 5), dtype=float)
    #Extract data from the DataFrame and store it in the param array
    for i in range(min(185, len(df))):  #Ensure the number of records does not exceed 185 
        for j in range(5):
            param[i, j] = float(df.iloc[i, j + 1])
    #‘CenterWavelengthNanometer’, ‘FullWidthAtHalfMaximumNanometer’,
    #‘SolarIrradianceWatt/Meter2/Micron’, ‘ReflectanceMulti’, ‘ReflectanceAdd’
    return param

# read meta data (txt)
def read_tfile(file):
    fileb = os.path.splitext(file)[0]  #Remove the file extension to obtain the base name 
    fileb = fileb + ".txt"  #Add “.txt” to the base name to create a new filename 
    csv_file = open(fileb, "r")  #Open the text file in read mode 
    record_list = []  #Initialize the list to store records 
    record = csv_file.readline()  #Start loading from the second line 
    while record :  #Loop until the end of the file 
        record_list.append(record.rstrip().split("="))  #Read each line, remove the newline character, split it using ‘=’ as the delimiter to create a list, and add it to record_list
        record = csv_file.readline()  #Read the next line 
    for record in record_list:  #Loop all records 
        if(record[0]=="RadianceMultiVNIR                                                      "):
            radiancemultivnir = float(record[1])
        if(record[0]=="RadianceAddVNIR                                                        "):
            radianceaddvnir = float(record[1])
        if(record[0]=="RadianceMultiSWIR                                                      "):
            radiancemultiswir = float(record[1])
        if(record[0]=="RadianceAddSWIR                                                        "):
            radianceaddswir = float(record[1])
    return radiancemultivnir, radiancemultiswir, radianceaddvnir, radianceaddswir

#Apply radiometric calibration to the input image 
def apply_radiometric(img, radmultivnir, radmultiswir, radaddvnir, radaddswir):
    im = np.ones([img.shape[0], img.shape[1]])  #Create a mask where all elements are set to 1 based on the image's height and width 
    # no data area
    im[img[:,:,10] == 0] = 0  # Set the mask to 0 for the position where band 10 is 0 in the image (indicating areas with no data)
    # change to float
    img = 1.0 * img  # Convert the input image to a floating-point number type
    # apply radiometric vnir
    for j in range(58):  # Apply radiometric calibration to bands 0 through 57 (VNIR region)
        img[:,:,j] = img[:,:,j] * radmultivnir + radaddvnir  # Multiply each band value by radmultivnir and add radaddvnir
    # apply radiometric swir
    for j in range(58, 185):  # Apply radiometric calibration to bands 58 through 184 (SWIR region)
        img[:,:,j] = img[:,:,j] * radmultiswir + radaddswir  # Multiply each band's value by radmultiswir and add radaddswir
    img[im == 0] = 0  # Set the image value to 0 for positions where the mask is 0 (areas with no data)
    return img

# Conversion from Pixel Space to Geospatial Space
def show_xy(src, x, y):
    width = src.RasterXSize #Get and store the width (number of columns) of the src raster dataset
    height = src.RasterYSize # Get and store the vertical height (number of rows) of the src raster dataset
    gt = src.GetGeoTransform() # Retrieve the geotransform (geographic transformation information) for the src raster dataset and store it in gt
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    X = gt[0] + x * gt[1] + y * gt[2]
    Y = gt[3] + x * gt[4] + y * gt[5]
    return X, Y

# Convert from geographic coordinates to latitude/longitude (WGS84)
#def show_latlon(src, x, y):
    old_cs= osr.SpatialReference() # Object containing the original coordinate system
    old_cs.ImportFromWkt(src.GetProjectionRef()) # Projection information obtained from the dataset (WKT format)
    # Define the WKT (Well-Known Text) representation of the WGS84 coordinate system as a string
    wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference() # Object for inserting a new coordinate system
    new_cs .ImportFromWkt(wgs84_wkt) # Import the defined WGS84 WKT string
    # Create an osr.CoordinateTransformation object to perform coordinate transformation from old_cs to new_cs, and store it in transform
    transform = osr.CoordinateTransformation(old_cs,new_cs)
    X, Y = show_xy(src, x, y) #Conversion from Pixel Space to Geospatial Space
    # Convert the calculated geographic coordinates X and Y to the WGS84 coordinate system (latitude and longitude) using transform, then store them
    latlong = transform.TransformPoint(X, Y)
    return latlong

# Function to extract and display specific bands from hyperspectral images
def get_rgb(img, b=8, g=18, r=28):
    ims = np.zeros([img.shape[0], img.shape[1], 3])  # Create a zero array with the image's height, width, and three RGB channels
    ims[:,:,0] = img[:,:,r]    #R
    ims[:,:,1] = img[:,:,g]    #G
    ims[:,:,2] = img[:,:,b]    #B
    max = np.max(ims)/3  # Get the maximum value in an image array
    ims /= max   # Divide the image by max and normalize
    ims = np.clip(ims, 0.0, 1.0)  # Clamp image array values to the range 0 to 255
    #The RGB values must be within the range of 0 to 1 for decimal values and 0 to 255 for integer values
    return ims
    
def show_img(img):
    fig, ax = plt.subplots()  
    im = ax.imshow(img)  
    plt.show()
    
#Extract SWIR data
def get_radiance(img, param, y, x):
    wave = param[58:185,0]
    rad = img[y, x, 58:185]
    list_data = [wave, rad]
    list_data_T = np.array(list_data).T
    return list_data_T

cols = [
    "CenterWavelengthNanometer",
    "FullWidthAtHalfMaximumNanometer",
    "SolarIrradianceWatt/Meter2/Micron",
    "ReflectanceMulti",
    "ReflectanceAdd",
]

df_param = pd.DataFrame(param, columns=cols)  
print(df_param)

# File loading
file = r"C:\Users\yudon\OneDrive\デスクトップ\メタン\2025_HISUI_1_地獄の門\HSHL1G_N402E0583_20210803092156_20240106123056\HSHL1G_N402E0583_20210803092156_20240106123056.tif"
img = tifffile.imread(file) # read tif file
param = read_bfile(file)    # read radiometric file (..._B.csv)
radmultivnir, radmultiswir, radaddvnir, radaddswir = read_tfile(file)  # read meta file (....txt)
img = apply_radiometric(img, radmultivnir, radmultiswir, radaddvnir, radaddswir)  # apply radiometric parameter
ims = get_rgb(img, b=8, g=18, r=28)
center = np.array([1410, 700]) #Center coordinates (y, x), change this
img_slice = img[center[0] - 100 : center[0] + 100, center[1] -100 : center[1] + 100, :] #Corrected image
ims_slice = ims[center[0] - 25 : center[0] + 25, center[1] -25 : center[1] + 25, :] #Image for display
show_img(ims)
show_img(ims_slice)

ch4_df = df_param[(df_param["CenterWavelengthNanometer"] >= 2248.0) &
               (df_param["CenterWavelengthNanometer"] <= 2298.0)].copy()
ch4_df = ch4_df.sort_values("CenterWavelengthNanometer")

H, W, C = img_slice.shape
out_A  = np.full((H, W), np.nan, dtype=np.float32)
out_C  = np.full((H, W), np.nan, dtype=np.float32)
out_QA = np.zeros((H, W), dtype=np.uint8)

df_param = df_param.copy()
df_param["absdiff"] = (df_param["CenterWavelengthNanometer"] - anchor_target).abs()
anchor_row = df_param.loc[df_param["absdiff"].idxmin()]

ANCHOR_CENTER = float(anchor_row["CenterWavelengthNanometer"])
ANCHOR_FWHM   = float(anchor_row["FullWidthAtHalfMaximumNanometer"])

CH4_BANDS = ch4_df[["CenterWavelengthNanometer","FullWidthAtHalfMaximumNanometer"]].to_numpy()

Lb_list = []
for center_nm, fwhm_nm in CH4_BANDS:
    Lb_list.append(
        convolve_band(wave_nm, rad_a10, float(center_nm), float(fwhm_nm))
    )
Lb_list = np.array(Lb_list, dtype=float)
N_CH4 = len(CH4_BANDS)  

for y in range(H):
    print(y)
    for x in range(W):
        data_rad = get_radiance(img_slice, param, y, x)
        df = pd.DataFrame(data_rad, columns=["Wavelength", "Radiance"])

        pix_wave = df["Wavelength"].to_numpy(dtype=float)
        pix_rad  = df["Radiance"].to_numpy(dtype=float)  

        idx_anchor  = int(np.abs(pix_wave - ANCHOR_CENTER).argmin())
        Lav_anchor  = float(pix_rad[idx_anchor])

        denom = (L10_anchor - L_path)
        if not np.isfinite(Lav_anchor) or denom <= 0:
            out_QA[y, x] = 1
            continue
        A = (Lav_anchor - L_path) / denom * 0.10
        if not np.isfinite(A) or A <= 0:
            out_QA[y, x] = 1
            continue

        chi = 0.10 / A

        sum_res = 0.0
        for j, (center_nm, fwhm_nm) in enumerate(CH4_BANDS):
            idx  = int(np.abs(pix_wave - center_nm).argmin())
            Lav  = float(pix_rad[idx])
            Lb   = float(Lb_list[j])
            Lres = Lb - chi * Lav
            sum_res += Lres

        C_scaled = (sum_res / (N_CH4 * A)) * 100.0

        out_A[y, x] = A
        out_C[y, x] = C_scaled
      
plt.imshow(out_C, cmap="jet"); plt.colorbar(); plt.title("C (scaled W m^-2 sr^-1 nm^-1)")

def c_to_ch4_ppm(C):
    return 0.000556 * (C**2) + 0.1519 * C + 1.4475

valid = np.isfinite(out_C)
if 'out_QA' in globals():
    valid &= (out_QA == 0)

CH4_ppm = np.full_like(out_C, np.nan, dtype=np.float32)
CH4_ppm[valid] = c_to_ch4_ppm(out_C[valid])

print("CH4(ppm) stats on valid pixels:",
      np.nanmin(CH4_ppm), np.nanmean(CH4_ppm), np.nanmax(CH4_ppm))

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(CH4_ppm, vmin=np.nanpercentile(CH4_ppm, 2), vmax=np.nanpercentile(CH4_ppm, 98))
plt.colorbar(label="CH$_4$ (ppm)")
plt.title("CH$_4$ map (from C)")
plt.axis("off")
plt.show()
