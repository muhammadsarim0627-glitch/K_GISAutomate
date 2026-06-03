import sys
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox

class Raster:
    def __init__(self, path):
        self.path = path
        self.ds = rasterio.open(path)
        self.profile = self.ds.profile

    def read(self):
        return self.ds.read(1).astype(float)

class Metadata:
    def __init__(self, path):
        self.data = {}
        with open(path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split(" = ")
                    self.data[k] = v.replace('"', "")

    def get(self, key):
        return float(self.data[key])

    def satellite(self):
        return self.data["SPACECRAFT_ID"]

class NDVI:
    def __init__(self, red_path, nir_path):
        self.red = Raster(red_path)
        self.nir = Raster(nir_path)

    def calculate(self):
        r = self.red.read()
        n = self.nir.read()
        ndvi = (n - r) / (n + r + 1e-10)
        profile = self.red.profile
        profile.update(dtype=rasterio.float32, count=1)
        return ndvi, profile

class NDWI:
    def __init__(self, green_path, nir_path):
        self.green = Raster(green_path)
        self.nir = Raster(nir_path)

    def calculate(self):
        g = self.green.read()
        n = self.nir.read()
        ndwi = (g - n) / (g + n + 1e-10)
        profile = self.green.profile
        profile.update(dtype=rasterio.float32, count=1)
        return ndwi, profile

class SAVI:
    def __init__(self, red_path, nir_path, L=0.5):
        self.red = Raster(red_path)
        self.nir = Raster(nir_path)
        self.L = L

    def calculate(self):
        r = self.red.read()
        n = self.nir.read()
        savi = ((n - r) / (n + r + self.L)) * (1 + self.L)
        profile = self.red.profile
        profile.update(dtype=rasterio.float32, count=1)
        return savi, profile

class NDBI:
    def __init__(self, swir_path, nir_path):
        self.swir = Raster(swir_path)
        self.nir = Raster(nir_path)

    def calculate(self):
        s = self.swir.read()
        n = self.nir.read()
        ndbi = (s - n) / (s + n + 1e-10)
        profile = self.swir.profile
        profile.update(dtype=rasterio.float32, count=1)
        return ndbi, profile

class RasterClipper:
    def __init__(self, raster_path, shp_path):
        self.raster = rasterio.open(raster_path)
        self.shp = gpd.read_file(shp_path)

    def clip(self):
        if self.shp.crs != self.raster.crs:
            self.shp = self.shp.to_crs(self.raster.crs)
        geom = [f["geometry"] for f in self.shp.__geo_interface__["features"]]
        clipped, transform = mask(self.raster, geom, crop=True)
        profile = self.raster.profile
        profile.update(height=clipped.shape[1], width=clipped.shape[2],
                       transform=transform, count=1, dtype=rasterio.float32)
        return clipped[0].astype(float), profile

class LST:
    def __init__(self, b10, b4, b5, mtl):
        self.thermal = Raster(b10)
        self.red = Raster(b4)
        self.nir = Raster(b5)
        self.meta = Metadata(mtl)
        self.ml = self.meta.get("RADIANCE_MULT_BAND_10")
        self.al = self.meta.get("RADIANCE_ADD_BAND_10")
        self.k1 = self.meta.get("K1_CONSTANT_BAND_10")
        self.k2 = self.meta.get("K2_CONSTANT_BAND_10")
        sat = self.meta.satellite()
        self.wavelength = 10.895e-6 if sat=="LANDSAT_8" else 10.9e-6
        self.rho = 1.438e-2

    def radiance(self):
        dn = self.thermal.read()
        dn[dn==0]=np.nan
        return self.ml*dn + self.al

    def brightness_temp(self):
        l = self.radiance()
        return self.k2 / np.log((self.k1 / l)+1)

    def emissivity(self):
        ndvi_tool = NDVI(self.red.path, self.nir.path)
        ndvi = ndvi_tool.calculate()[0]
        ndvi_min = np.nanmin(ndvi)
        ndvi_max = np.nanmax(ndvi)
        pv = ((ndvi - ndvi_min)/(ndvi_max - ndvi_min + 1e-10))**2
        e = 0.004*pv + 0.986
        e = np.where(e<=0,0.986,e)
        return e

    def calculate(self):
        tb = self.brightness_temp()
        e = self.emissivity()
        lst = tb / (1 + (self.wavelength * tb / self.rho) * np.log(e))
        lst = np.where(np.isnan(lst)|np.isinf(lst), np.nan, lst)
        r = self.red.read()
        n = self.nir.read()
        ndvi = (n - r)/(n + r + 1e-10)
        lst = np.where(ndvi<0.1,np.nan,lst)
        lst_c = (lst - 273.15)*0.96
        profile = self.thermal.profile
        profile.update(dtype=rasterio.float32, count=1)
        return lst_c, profile

class DEMAnalysis:
    def __init__(self, dem_path):
        self.dem = Raster(dem_path)

    def slope_aspect(self):
        dem = self.dem.read()
        x, y = np.gradient(dem)
        slope = np.arctan(np.sqrt(x**2 + y**2)) * (180/np.pi)
        aspect = np.arctan2(-x, y) * (180/np.pi)
        profile = self.dem.profile
        profile.update(dtype=rasterio.float32, count=1)
        return slope.astype(np.float32), aspect.astype(np.float32), profile

class BandComposite:
    def __init__(self, band_paths):
        self.bands = [Raster(p) for p in band_paths]

    def composite(self):
        profile = self.bands[0].profile
        profile.update(count=len(self.bands), dtype=rasterio.float32)
        data = np.stack([b.read() for b in self.bands])
        return data.astype(np.float32), profile

class GISApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K_GISAutomate")
        self.setGeometry(100,50,1000,700)
        y=50
        for name, func in [("NDVI", self.run_ndvi),
                           ("NDWI", self.run_ndwi),
                           ("SAVI", self.run_savi),
                           ("NDBI", self.run_ndbi),
                           ("Clip Raster", self.run_clip),
                           ("LST", self.run_lst),
                           ("DEM Slope/Aspect", self.run_dem),
                           ("Composite Bands", self.run_composite)]:
            btn=QPushButton(name,self)
            btn.setGeometry(20,y,180,40)
            btn.clicked.connect(func)
            y+=50

    def run_ndvi(self):
        red,_=QFileDialog.getOpenFileName(self,"Band 4","","GeoTIFF (*.tif)")
        nir,_=QFileDialog.getOpenFileName(self,"Band 5","","GeoTIFF (*.tif)")
        ndvi, profile=NDVI(red,nir).calculate()
        plt.imshow(ndvi, cmap="RdYlGn");plt.colorbar();plt.title("NDVI");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save NDVI","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(ndvi,1)

    def run_ndwi(self):
        green,_=QFileDialog.getOpenFileName(self,"Green Band","","GeoTIFF (*.tif)")
        nir,_=QFileDialog.getOpenFileName(self,"NIR Band","","GeoTIFF (*.tif)")
        ndwi, profile=NDWI(green,nir).calculate()
        plt.imshow(ndwi,cmap="Blues");plt.colorbar();plt.title("NDWI");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save NDWI","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(ndwi,1)

    def run_savi(self):
        red,_=QFileDialog.getOpenFileName(self,"Red Band","","GeoTIFF (*.tif)")
        nir,_=QFileDialog.getOpenFileName(self,"NIR Band","","GeoTIFF (*.tif)")
        savi, profile=SAVI(red,nir,L=0.5).calculate()
        plt.imshow(savi,cmap="Greens");plt.colorbar();plt.title("SAVI");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save SAVI","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(savi,1)

    def run_ndbi(self):
        swir,_=QFileDialog.getOpenFileName(self,"SWIR Band","","GeoTIFF (*.tif)")
        nir,_=QFileDialog.getOpenFileName(self,"NIR Band","","GeoTIFF (*.tif)")
        ndbi, profile=NDBI(swir,nir).calculate()
        plt.imshow(ndbi,cmap="Reds");plt.colorbar();plt.title("NDBI");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save NDBI","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(ndbi,1)

    def run_clip(self):
        raster,_=QFileDialog.getOpenFileName(self,"Raster","","GeoTIFF (*.tif)")
        while True:
            shp,_=QFileDialog.getOpenFileName(self,"Shapefile","","Shapefile (*.shp)")
            try:
                clipped,profile=RasterClipper(raster,shp).clip()
                break
            except:
                again=QMessageBox.question(self,"No Overlap","Select another shapefile?",QMessageBox.Yes|QMessageBox.No)
                if again==QMessageBox.No:return
        plt.imshow(clipped,cmap="gray");plt.title("Clipped Raster");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save Clipped","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(clipped,1)

    def run_lst(self):
        b10,_=QFileDialog.getOpenFileName(self,"Band 10","","GeoTIFF (*.tif)")
        b4,_=QFileDialog.getOpenFileName(self,"Band 4","","GeoTIFF (*.tif)")
        b5,_=QFileDialog.getOpenFileName(self,"Band 5","","GeoTIFF (*.tif)")
        mtl,_=QFileDialog.getOpenFileName(self,"MTL File","","MTL (*.txt)")
        lst_tool=LST(b10,b4,b5,mtl)
        lst, profile=lst_tool.calculate()
        plt.imshow(lst,cmap="inferno");plt.colorbar(label="°C");plt.title("LST");plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save LST","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(lst,1)

    def run_dem(self):
        dem,_=QFileDialog.getOpenFileName(self,"DEM Raster","","GeoTIFF (*.tif)")
        slope,aspect,profile=DEMAnalysis(dem).slope_aspect()
        plt.subplot(1,2,1);plt.imshow(slope,cmap="terrain");plt.title("Slope");plt.colorbar()
        plt.subplot(1,2,2);plt.imshow(aspect,cmap="hsv");plt.title("Aspect");plt.colorbar()
        plt.show()
        save,_=QFileDialog.getSaveFileName(self,"Save Slope","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(slope,1)
        save,_=QFileDialog.getSaveFileName(self,"Save Aspect","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst: dst.write(aspect,1)

    def run_composite(self):
        files,_=QFileDialog.getOpenFileNames(self,"Select Bands 1-7","","GeoTIFF (*.tif)")
        composite, profile=BandComposite(files).composite()
        save,_=QFileDialog.getSaveFileName(self,"Save Composite","","GeoTIFF (*.tif)")
        if save: 
            with rasterio.open(save,"w",**profile) as dst:
                dst.write(composite)

app=QApplication(sys.argv)
window=GISApp()
window.show()
sys.exit(app.exec_())