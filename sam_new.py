import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

#os.environ["PROJ_LIB"] = "C:/projects2025/wdod_ZSW/.pixi/envs/default/Library/share/proj"
#os.environ["GDAL_DRIVER_PATH"] ="C:/projects2025/wdod_ZSW/.pixi/envs/default/Library/lib/gdalplugins"
#os.environ["PATH"] ="C:/projects2025/wdod_ZSW/.pixi/envs/default/Library/bin"

import numpy as np
import torch
import sys
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal, ogr
import gc
import json 
import rasterio.features
from rasterio.enums import Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, LineString,Polygon, MultiLineString,Point
from shapely.ops import polygonize
from shapely.ops import linemerge
from  arg_coords import  getPointlist
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import base64
import requests
from shapely.validation import make_valid
from PIL import Image
from  io import BytesIO
import io
import time
from PIL import Image


# configuration

sam2_checkpoint = "C:/Users/Tjip.vanDale/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

items=0
debugging=0
pointList=getPointlist()
#print (pointList)
procesDir='C:/projects2025/proces/test'
file2proces=procesDir+'/h&a_totaal.tif'
file2proces="F:/HenA/VoorjaarsVlucht2025POC.tif"
file2proces=procesDir+'/h&a_voor_demping.tif'

ndivFile= "ndvi.tif"

gpkgMasks='masker_sam.gpkg'
gpkgLayer='masks_sam'

gpkgMasks_ai='masker_ai.gpkg'
gpkgLayer_ai='masks_ai'

gpkgWaterdeel="waterdelen.gpkg"
gpkgWaterdeel_Layer="bgt_vlakken"

gpkgWaterdeelStatus="waterdeel_status.gpkg"
gpkgWaterdeelStatus_layer="waterdeel_status"

sieve_size=500
crop_size=20

bgtSamAfwijkingsPerc=10
watervalNdvi=0.07


#url_api="http://192.168.1.100:8080/post_example"


#ithax
 # ITHAX API Url:
api_url="https://demo.ithax.ai:9001/ithax_controle_agent"

    # Defaults
crs = "EPSG:28992"
resolution = 0.08
imageformat = "image/jpg"
version = "1.3.0"
wms_getcapabilities_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?request=GetCapabilities"
layers = "Actueel_orthoHR"
#ithax    
#configuration

pointList2proces=[[2,3]]
    
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
 
#device = torch.device("cpu")
print(f"using device: {device}")
print(f"using geopkg: {gpkgMasks}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook$length / 2.0.
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

    np.random.seed(3)
  # %%  


def IsPointInwater(CentrePoint):

    pixel_coord=rdCoord2LocalpixelCoord (f"{procesDir}/{ndivFile}", CentrePoint)
    image = Image.open(f"{procesDir}/{ndivFile}")

    # Get pixel value at (x, y)
    pixel_value = image.getpixel(pixel_coord)

    if  debugging==1:
        print(pixel_value)

    if pixel_value<watervalNdvi :
        return 1
    else:
        return 0


def getBGTIntsectieSam(centerpoint, pointno):

    xmin = centerpoint[0]-crop_size
    ymin = centerpoint[1]-crop_size
    xmax = centerpoint[0]+crop_size
    ymax = centerpoint[1]+crop_size

    #onderzoeksgebied SAM
    proces_area = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    try:     
        gdf_bgt_polygons=gpd.read_file(f"{procesDir}/{gpkgWaterdeel}", layer=gpkgWaterdeel_Layer)
            
        centerpoint_proces_area = Point(centerpoint[0], centerpoint[1])
    
        # bgt area that is covering the selection point
        selected_polygon_BGT = gdf_bgt_polygons[gdf_bgt_polygons.contains(centerpoint_proces_area)]
         
        # Compute intersection bgt_waterdeel with procesarea and force to 2d
        bgt_intersectie_2proces = selected_polygon_BGT.intersection(proces_area,align=False)
        bgt_intersectie_2proces_2d=bgt_intersectie_2proces.force_2d()

        gdf_bgt_intersectie_from_serie = gpd.GeoDataFrame(geometry=bgt_intersectie_2proces_2d)
        
        gdf_bgt_intersectie_from_serie
        if gdf_bgt_intersectie_from_serie.is_empty.all():
            return False,gdf_bgt_intersectie_from_serie

        if "point_no" not in gdf_bgt_intersectie_from_serie.columns:
            gdf_bgt_intersectie_from_serie["point_no"] = pointno 

        if debugging==1:
            gdf_bgt_intersectie_from_serie.to_file(f"{procesDir}/bgtvlaktest", layer="bgt_vlak", driver="GPKG")

        return True, gdf_bgt_intersectie_from_serie
    
    except Exception as e:
        print(f"getBGTIntsectieSam failed ({e}).")
        return False, 0



def getBGTSamIthaxDiffProcessArea(dfBGTPolygon, df_samPolygon, df_Diff_Ithax, point_no):
    
    try:
        total_area_sam = df_samPolygon.geometry.area.sum()
        total_area_BGT = dfBGTPolygon.geometry.area.sum()
        total_area_Ithax = df_Diff_Ithax.geometry.area.sum()
        total_diff=total_area_BGT- (total_area_sam+  total_area_Ithax)
        total_diff_perc= total_diff/total_area_BGT *100

        print(f"total_area_sam: {total_area_sam}; total_area_BGT: {total_area_BGT}; total_area_Ithax: {total_area_Ithax}; Verschil: {total_diff}; Perc: {total_diff_perc}")

        if abs(total_diff_perc)>bgtSamAfwijkingsPerc:
              return True , 2, total_diff_perc
        else:
             return True, 1, total_diff_perc
        
    except Exception as e:
        print(f"getBGTSamIthaxDiffProcessArea failed ({e}).")
        return False, 0
     

def setStatusSAMPolygon( pointno,status,BGT_SAM_Diff_Perc,BGT_SAM_Ithax_Diff_Perc,bgt_intersectie_2proces_2d):
    #Sammasks
    gdf_mask_polygon = gpd.read_file(f"{procesDir}/{gpkgMasks}", layer=gpkgLayer)
    gdf_mask_polygon.loc[gdf_mask_polygon['point_no'] == pointno, 'status'] = status
    gdf_mask_polygon.to_file(f"{procesDir}/{gpkgMasks}", layer=gpkgLayer, driver="GPKG") 

    if "status" not in bgt_intersectie_2proces_2d.columns:
            bgt_intersectie_2proces_2d["status"] = status 

    if "diff_perc_bgt_sam" not in bgt_intersectie_2proces_2d.columns:
            bgt_intersectie_2proces_2d["diff_perc_bgt_sam"] = 0.0
            bgt_intersectie_2proces_2d["diff_perc_bgt_sam"] = BGT_SAM_Diff_Perc

    if "diff_perc_bgt_ithax_sam" not in bgt_intersectie_2proces_2d.columns:
            bgt_intersectie_2proces_2d["diff_perc_bgt_ithax_sam"] = 0.0  
            bgt_intersectie_2proces_2d["diff_perc_bgt_ithax_sam"] = BGT_SAM_Ithax_Diff_Perc  

    bgt_intersectie_2proces_2d.to_file(f"{procesDir}/{gpkgWaterdeelStatus}", layer=gpkgWaterdeelStatus_layer, driver="GPKG",mode='a') 
    
    return True


def getBGTSAMDifferenceProcessArea(centerpoint):
    
    xmin = centerpoint[0]-crop_size
    ymin = centerpoint[1]-crop_size
    xmax = centerpoint[0]+crop_size
    ymax = centerpoint[1]+crop_size

    #onderzoeksgebied SAM
    proces_area = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    afwijking2big=False
    selected_polygon_SAM=1
    BGT_SAM_Diff_Perc=0

    # to do aanpassen tvd

    try:
        #BGT
        #gdf_bgt_polygons = gpd.read_file("C:\\projects2025\\h&a\\Innovatie_1\\data\\waterdelen.gpkg", layer="bgt_vlakken")
        gdf_bgt_polygons=gpd.read_file(f"{procesDir}/{gpkgWaterdeel}", layer=gpkgWaterdeel_Layer)
        
        centerpoint_proces_area = Point(centerpoint[0], centerpoint[1])
        # select the sam and bgt area that is covering the selection point
        selected_polygon_BGT = gdf_bgt_polygons[gdf_bgt_polygons.contains(centerpoint_proces_area)]

        # Compute intersection bgt_waterdeel and force to 2d
        bgt_intersectie_2proces = selected_polygon_BGT.intersection(proces_area,align=False)
        bgt_intersectie_2proces_2d=bgt_intersectie_2proces.force_2d()

          #Sammasks
        gdf_mask_polygon = gpd.read_file(f"{procesDir}/{gpkgMasks}", layer=gpkgLayer)

     
        #Be sure that the sam crs is used
        gdf_mask_polygon.crs=gdf_bgt_polygons.crs
        gdf_mask_polygon.to_crs=gdf_bgt_polygons.crs

        

        selected_polygon_SAM_all = gdf_mask_polygon[gdf_mask_polygon.contains(centerpoint_proces_area)]
        selected_polygon_SAM = selected_polygon_SAM_all.iloc[[0]]
         
        # Compute intersection sam with procesarea 
        sam_intersectie_2proces = selected_polygon_SAM.intersection(proces_area,align=False)
        sam_intersectie_2proces_2d=sam_intersectie_2proces.force_2d()

        #tvd niet nodig
        # gpkgMasks_SAM_intersectPath= f"{procesDir}/{gpkgMasks_SAM_intersect}"
        # sam_intersectie_2proces_2d.to_file(gpkgMasks_SAM_intersectPath, layer=gpkgMasks_SAM_intersectLayer, driver="GPKG", mode='a') 
 
        
        areaDiff=bgt_intersectie_2proces_2d.geometry.area.sum()-selected_polygon_SAM.geometry.area.sum()
        areaTotal=bgt_intersectie_2proces_2d.geometry.area.sum() +selected_polygon_SAM.area.sum()
        # areaDiff=bgt_intersectie_2proces_2d.geometry.area.values[0]-selected_polygon_SAM.area.values[0] 
        # areaTotal=bgt_intersectie_2proces_2d.geometry.area.values[0]+selected_polygon_SAM.area.values[0]
        BGT_SAM_Diff_Perc=areaDiff/bgt_intersectie_2proces_2d.geometry.area.sum()*100

        print (f" Verschil% BGT SAM :{BGT_SAM_Diff_Perc}")
        
        
        if BGT_SAM_Diff_Perc>bgtSamAfwijkingsPerc:
            afwijking2big=True
        else: 
            afwijking2big=False
        return True,  afwijking2big, bgt_intersectie_2proces_2d, selected_polygon_SAM, BGT_SAM_Diff_Perc
    
    except Exception as e:
        print(f"getBGTIntersectionForProcessArea failed ({e}).")
    
    return False,afwijking2big, bgt_intersectie_2proces_2d, selected_polygon_SAM,BGT_SAM_Diff_Perc
    

def postprocesIthax(centerpoint,dfMasks,gdf_bgt_polygons,pointno):
    gpkgMasks_ithax='masker_ithax.gpkg'
    gpkgLayer_ithax='masks_ithax'

    xmin = centerpoint[0]-crop_size
    ymin = centerpoint[1]-crop_size
    xmax = centerpoint[0]+crop_size
    ymax = centerpoint[1]+crop_size

    #onderzoeksgebied SAM
    proces_area = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    
    try:
        #BGT
        #gdf_bgt_polygons = gpd.read_file("C:\\projects2025\\h&a\\Innovatie_1\\data\\waterdelen.gpkg", layer="bgt_vlakken")
        #gdf_bgt_polygons=gpd.read_file(f"{procesDir}/{gpkgWaterdeel}", layer=gpkgWaterdeel_Layer)
        
        #Sammasks
        gdf_mask_polygon = gpd.read_file(f"{procesDir}/{gpkgMasks}", layer=gpkgLayer)
        #iTHAX Masks
        gdf_mask_ITHAX_polygon = dfMasks 

        centerpoint_proces_area = Point(centerpoint[0], centerpoint[1])

        #Be sure that the sam crs is used
        gdf_mask_polygon.crs=gdf_bgt_polygons.crs
        gdf_mask_polygon.to_crs=gdf_bgt_polygons.crs

        # select the sam and bgt area that is covering the selection point
        #selected_polygon_BGT = gdf_bgt_polygons[gdf_bgt_polygons.contains(centerpoint_proces_area)]
        selected_polygon_SAM_all = gdf_mask_polygon[gdf_mask_polygon.contains(centerpoint_proces_area)]
        #there can be more so select the first one
        selected_polygon_SAM = selected_polygon_SAM_all.iloc[[0]]

        # Compute intersection bgt_waterdeel and force to 2d
        #bgt_intersectie_2proces = gdf_bgt_polygons.intersection(proces_area,align=False)
        
        #bgt_intersectie_2proces_2d=bgt_intersectie_2proces.force_2d()

        #Be sure that for ITHAX the sam crs is used
        gdf_mask_ITHAX_polygon = gdf_mask_ITHAX_polygon.to_crs(gdf_mask_polygon.crs)
        
        # clip  ITHAX Maskers with bgt intersection
        #clipped_mask_ITHAX_polygon = gdf_mask_ITHAX_polygon.clip(bgt_intersectie_2proces_2d)
        clipped_mask_ITHAX_polygon = gdf_mask_ITHAX_polygon.clip(gdf_bgt_polygons)

        #store clipped masks
        gpkgMasksFile_ithax= f"{procesDir}/{gpkgMasks_ithax}"
        #clipped_mask_ITHAX_polygon.to_file(gpkgMasksFile_ithax, layer="testith", driver="GPKG", mode='a') 

        #remove part found by sam
        #bgt_intersectie_2proces = selected_polygon_BGT.intersection(proces_area,align=True)

        selected_polygon_SAM.crs=gdf_bgt_polygons.crs
        selected_polygon_SAM.to_crs=gdf_bgt_polygons.crs

        #selected_polygon_SAM.to_file(procesDir+"/tst.gpkg", layer="test", driver="GPKG") 
        
        diff= clipped_mask_ITHAX_polygon.geometry.apply(lambda geom: geom.difference(selected_polygon_SAM.geometry.iloc[0]))
        
        gdf_diff_from_serie = gpd.GeoDataFrame(geometry=diff)
    
        if "point_no" not in gdf_diff_from_serie.columns:
            gdf_diff_from_serie["point_no"] = pointno 
        
    
        #store clipped masks
        gpkgMasksFile_ithax= f"{procesDir}/{gpkgMasks_ithax}"

        
        gdf_diff_from_serie.to_file(gpkgMasksFile_ithax, layer=gpkgLayer_ithax, driver="GPKG", mode='a') 

        return gdf_diff_from_serie
    
        if debugging==1:
            #Plot results
            fig, ax = plt.subplots(figsize=(8, 8))
            #selected_polygon_SAM.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5, label='Layer 1')
            #bgt_intersectie_2proces.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, label='Layer 2')
            gdf.plot(ax=ax, edgecolor='blue', facecolor='red', linewidth=2, label='Layer 3')
            #clipped_mask_ITHAX_polygon.plot(ax=ax, edgecolor='green', facecolor='yellow', linewidth=2, label='Layer 3')
        
            plt.title("Ithax Polygons Processed")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend
            plt.show()
    except Exception as e:
        print(f"postprocesIthax failed ({e}).")



def getMaskFromIthax( centerpoint,pointno):

    print(f"Getting masks from Ithax...for pointNo.:{pointno}")

    #with open(subset_file, "rb") as image_file:
               # base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    # Send request to API server
    #response_data = requests.post(url_api, json={"image_base64": base64_image})
    #json_data = response_data.json()

    #mask=json_data["received"] 
    #=============


    xmin = centerpoint[0]-crop_size
    ymin = centerpoint[1]-crop_size
    xmax = centerpoint[0]+crop_size
    ymax = centerpoint[1]+crop_size

    output_geojson_file = (f"./ithax_{pointno}.geojson")

    payload = {
        "wms_getcapabilities": wms_getcapabilities_url,
        "layers": layers,
        "version": version,
        "crs": crs,
        "format": imageformat,
        "resolution": resolution,
        "boundingbox": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }
    }
           
    for attempt in range(2):  # 0 = first attempt, 1 = retry
        try:
            response = requests.post(api_url, json=payload, timeout=600)

            print(f"Ithax req,:{payload}") 
 
            # Check whether there is a correct response
            if response.status_code == 200:
                result = response.json()
                
                if  result["features"]:
                    
                    jsonstring=json.dumps(result)
                    print( jsonstring)
                    
                    # Save as GeoJSON file
                    with open(output_geojson_file, 'w') as f:
                        json.dump(result, f, indent=2)

                    # Print and return geojson file location              
                    print(f"✅ Success!") 

                    # Save to GeoPackage
                    geojson_bytes = io.BytesIO(jsonstring.encode('utf-8'))

                    # Read it with geopandas
                    gdfMasks = gpd.read_file(geojson_bytes)
                        #add column 2 store pointno
                    if "point_no" not in gdfMasks.columns:
                        gdfMasks["point_no"] = pointno
                        
                    geopackage_path = procesDir+"/"+gpkgMasks_ai      
                    gdfMasks.to_file(geopackage_path, layer=gpkgLayer_ai, driver="GPKG",  mode='a')
        
                    return True, gdfMasks
                else:  
                    print("No data from ithax")
                    gdfMasks = None
                    return False ,gdfMasks
            else:
                if attempt == 0:  # First attempt failed
                    print(f"Request failed (Status {response.status_code}), retrying in 1 second...")
                    time.sleep(5)
                    continue
                else:  # Second attempt failed
                    print(f"ITHAX API Error {response.status_code}: {response.text}")
                    return False #output_geojson_file
                    
        except Exception as e:
            if attempt == 0:  # First attempt failed
                print(f"Request failed ({e}), retrying in 1 second...")
                time.sleep(1)
                continue
            else:  # Second attempt failed
                print(f"Error: {e}")
                return False#output_geojson_file
    
    return output_geojson_file

def storeMask(mask,subset_file):
     #no longer used
     with rasterio.open(subset_file) as src:
        data = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
      
        print(transform.translation)  # X/Y shift
        print(transform.scale)  # Pixel size

        # Generate a binary mask of the valid data pixels
        #mask = np.ma.masked_invalid(data).mask.astype(int)
        
        # Decode Base64 to bytes
        image_data = base64.b64decode(mask)

        # Convert bytes to an image using Pillow
        image = Image.open(BytesIO(image_data))

        # Convert image to NumPy array
        image_array = np.array(image)

        shapes = rasterio.features.shapes(image_array, transform=transform)
         
        for polygon, value in shapes:
            if value == 255:                
                boundary = shape(polygon).boundary
                
                polygons =gpd.GeoSeries(boundary)
               
                df = gpd.GeoDataFrame(geometry=polygons, crs=crs)

                boundary_geom = df.geometry.union_all()  # Merge all line segments into one
                 #multilinestring.convex_hull
               
                if boundary_geom.geom_type == "MultiLineString":
                    coords = [coord for line in boundary_geom.geoms for coord in line.coords]
                else :
                    coords = list(boundary_geom.coords) 

                polygon = Polygon(coords)
                outer_boundary = polygon.exterior
                coords = [coord for coord in outer_boundary.coords]
                polygon = Polygon(coords)
                # remove kickbacks
                polygon2=make_valid(polygon.buffer(0))
                gdf = gpd.GeoDataFrame({"geometry": [polygon2]}, crs=crs)
               
            
                # Append the GeoDataFrame to the masks geopackage
                maskfileGPKGAI=procesDir+ "//" + gpkgMasks_ai 

        gdf.to_file(maskfileGPKGAI, layer=gpkgLayer_ai, driver="GPKG", mode='a') 

        return gdf

def readLastProcessedItemFile():
    # Open the file in read mode
    with open( procesDir+'//lastProcessedItem.txt',  'r') as file:
        # Read the contents of the file
        content = file.read()
        global items
        items= int(content)
        
        pointArrayLength= len(pointList)
        start2proces=pointArrayLength-items
        
        if start2proces==0:
            pointList2proces=[[1,1]]
        else:
            pointList2proces=pointList[-start2proces:]
 
        file.close()
        return pointList2proces
# %%
def writeLastProcessedItemFile(itemProcessed):
    with open(  procesDir+'//lastProcessedItem.txt', 'w') as file:
        file.write(itemProcessed)
        file.close()    

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        break

def rdCoord2LocalpixelCoord (subsetfile, centerPoint):
    
    print('Recalculate RD to local Pixel Coordinate')
    # Opens source dataset
    src_ds = gdal.Open(subsetfile) 
   
    geoTransform = src_ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * src_ds.RasterXSize
    miny = maxy + geoTransform[5] * src_ds.RasterYSize
    
    #resolution
    x_res = (maxx- minx)   /src_ds.RasterXSize   # 426  #tvd
    y_res =  (maxy- miny) /src_ds.RasterYSize #316
    
    #print ( 'Bron resolutie x_res: {} y_res: {}'.format(x_res,y_res))
   
    factor=1/ x_res 
    
    pixelX=(centerPoint[0]-minx)*factor
    pixelY=(maxy-centerPoint[1])*factor
     
    upper_left = [minx, maxy]
    bottom_left = [minx, miny]
    upper_right = [maxx, maxy]
    bottom_right = [maxx, miny]
    
    src_ds = None
    
    pixelCoord=[pixelX,pixelY]
    return  pixelCoord

def subset_for_sam(coordinate, file2Segment):
    
    print('Subset for predictor')
    centerPoint= coordinate
    print (centerPoint)
    outputBoundsARR=[centerPoint[0]-crop_size,centerPoint[1]-crop_size ,centerPoint[0]+crop_size,centerPoint[1]+crop_size]
    fname_subset= procesDir+'//subset.tif'
    ds= gdal.Open(file2Segment)
    #g = gdal.Warp(fname_out, ds, format="GTiff", cutlineDSName=selectie, cutlineLayer = 'silogroepselectie', cropToCutline=True, dstNodata=0)
    g = gdal.Warp(fname_subset, ds, format="GTiff", outputBounds=outputBoundsARR, cropToCutline=True, dstNodata=0)
    print('Subset  succesvol') 

    return fname_subset 
    
def multiline_to_linestring(geometry):
    if isinstance(geometry, MultiLineString):
        return linemerge(geometry)
    return geometry

def generate_vector_from_masks(mask, subsetfile, pointno,isWater):
    
    lines = []
    
    with rasterio.open(subsetfile) as src:
        data = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
        #print (crs)
        
        # Generate a binary mask of the valid data pixels
        #mask = np.ma.masked_invalid(data).mask.astype(int)
        
        mf = (mask * 255).astype(np.uint8)
        mf_sieved=rasterio.features.sieve(mf, sieve_size, out=None, mask=None, connectivity=8)
        
        with rasterio.open("c:\\temp\\sammask.tif",
            "w",
            driver="GTiff",
            height=mf_sieved.shape[0],
            width=mf_sieved.shape[1],
            count=1,
            dtype=mask.dtype,
            transform=transform,
        ) as dst:
            dst.write(mf_sieved, 1)

       
        # Generate polygon geometries from the valid data mask
        shapes = rasterio.features.shapes(mf_sieved, transform=transform)
    
        index=1
        print( len(mf))
        for polygon, value in shapes:
            if value == 255:                
                #df = gpd.GeoDataFrame(geometry=polygon, crs=crs)
                boundary = shape(polygon).boundary
                poly = ogr.CreateGeometryFromJson(json.dumps(polygon))      
                area=(poly.GetArea())
                 
                print (f"area{area}") 

                index=index+1
                if area>10:
                    if isinstance(boundary, MultiLineString):
                        lines.append(multiline_to_linestring(boundary) )
                    if isinstance(boundary, LineString):
                        lines.append(boundary )
                    
    # Create a GeoDataFrame from the polylines

    polygons =gpd.GeoSeries(polygonize(lines))
    df = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    #add column 2 store pointno
    if "point_no" not in df.columns:   
        df["point_no"] = pointno
    
    #is it water or not based on ndvi
    if "iswater" not in df.columns:  
      df["iswater"] = isWater
     


    # Append the GeoDataFrame to the masks geopackage
    maskfileGPKG=procesDir+ "//" + gpkgMasks 
    df.to_file(maskfileGPKG, layer=gpkgLayer, driver="GPKG", mode='a') 
    src.closed    

    print('Conversie naar shape  succesvol')
    
    return maskfileGPKG

def startProcessingBatch():
    pointList2proces=readLastProcessedItemFile()
    print ('Running in batchmode')
    #pointList=[  [257401 ,590459]]
    aantal=len(pointList2proces) 

    count=0
    result=False

    for x,y   in pointList2proces:
        if x!=1:     
            centerPoint= [x,y]
            count=count+1

           


            result,df_bgt_intersectie_procesArea= getBGTIntsectieSam(centerPoint,items+count)
            if result==False:
                print(f"ProcesPoint not in BGT vlak: {items+count} ,{centerPoint}" )
            else:    
                procesImage( centerPoint,  items+count)
                print ( 'Plot: {} van {} verwerkt'.format(count,aantal ))
                totalprocessed = items+count
                writeLastProcessedItemFile(str(totalprocessed))
                if count==10:
                    print('GPU Mem. usage exceeded, restart kernel')
                    break
   
    print('Batch End')  
   
def procesImage(centerPoint,pointno):
    result=False
    
    
    # get intersectie of procesarea with bgt

    file4sam= subset_for_sam(centerPoint,file2proces)

    image = Image.open(file4sam)
    image = np.array(image.convert("RGB"))
    if debugging==1:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    
    pixelCoord=rdCoord2LocalpixelCoord (file4sam, centerPoint)
    
    input_point = np.array([[pixelCoord[0], pixelCoord[1]]])
    input_label = np.array([1])
    if debugging==0:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()  
    
    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)
      
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
       
    masks.shape  # (number_of_masks) x H x W
    
    if debugging==0:
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    
    iswater=IsPointInwater(centerPoint)
    #store the results in geopackage
    df_samPolygon2=generate_vector_from_masks(masks[0], file4sam,pointno, iswater) # to do
 
    # BGT intersection in procesarea
    result,df_bgt_intersectie_procesArea= getBGTIntsectieSam(centerPoint,pointno)
    if result==False:
          print(f"ProcesPoint not in BGT vlak: {pointno} ,{centerPoint}" )

    #  get difference between bgt and sam 
    BGT_SAM_Diff_Perc=0   
    BGT_SAM_Ithax_Diff_Perc=0
    
    if result==True:
        result,afwijking2big,df_BGTPolygon, df_samPolygon,BGT_SAM_Diff_Perc  = getBGTSAMDifferenceProcessArea(centerPoint)

    if result==True: 
        afwijking2big=False # weghalen tvd 
        if (afwijking2big==True):
         #difference too big, find cause via  Ithax qualification  
            resultIthax, dfMasksIthax=getMaskFromIthax(centerPoint,pointno)
            if resultIthax==False:
                print(f"Ithax class. failed for: {pointno} ,{centerPoint}" )
                setStatusSAMPolygon(pointno,"nok",BGT_SAM_Diff_Perc,0,df_bgt_intersectie_procesArea)    
            else:
                #select intersectie Ithx with bgt and difference with sam 
                df_Masks_Ithax_diff=postprocesIthax(centerPoint,dfMasksIthax,df_bgt_intersectie_procesArea,pointno)
                print(f"Ithax class. success for: {pointno} ,{centerPoint}" )
                
                # get difference between sam and Ithax   
                resultSamIthax, action, BGT_SAM_Ithax_Diff_Perc=getBGTSamIthaxDiffProcessArea(df_BGTPolygon, df_samPolygon, df_Masks_Ithax_diff,pointno)
                if resultSamIthax==True: 
                    if action==1:  
                        setStatusSAMPolygon(pointno,"ok",BGT_SAM_Diff_Perc ,BGT_SAM_Ithax_Diff_Perc,df_bgt_intersectie_procesArea)
                    else:
                        setStatusSAMPolygon(pointno,"nok",BGT_SAM_Diff_Perc,BGT_SAM_Ithax_Diff_Perc,df_bgt_intersectie_procesArea)
            #no objects found by ithax therefore zero difference and           
           
        else:
             if BGT_SAM_Diff_Perc<=bgtSamAfwijkingsPerc:
                setStatusSAMPolygon(pointno,"ok",BGT_SAM_Diff_Perc,0,df_bgt_intersectie_procesArea)
             else:
                 setStatusSAMPolygon(pointno,"nok",BGT_SAM_Diff_Perc,0,df_bgt_intersectie_procesArea)
             #setStatusSAMPolygon(pointno,"ok",BGT_SAM_Diff_Perc ,BGT_SAM_Ithax_Diff_Perc,df_bgt_intersectie_procesArea)
    else:
    
       setStatusSAMPolygon(pointno,"NAN",BGT_SAM_Diff_Perc,BGT_SAM_Ithax_Diff_Perc,df_bgt_intersectie_procesArea)

    #clean, release & close
    collected =gc.collect()
    print(f"Garbage collector: collected {collected} objects.")
    del(masks)
    del predictor 
    torch.cuda.empty_cache()
    

#entry Point    
startProcessingBatch()
print("Restart Kernel to free GPU Mem")

os._exit(00)


