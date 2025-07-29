# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:23:48 2025

@author: Tjip.vanDale
"""

# Set the layer name and attribute name
layer_name = 'points2proces'
attribute_name = 'xy_string'

# Get the layer
layer = QgsProject.instance().mapLayersByName(layer_name)[0]

count = layer.selectedFeatureCount()
#count = layer.featureCount()

print (count)
counter=0
# Open a text file to write the attribute values
output_file = 'C:/Users/Tjip.vanDale/segment-anything-2/arg_coords.py'
with open(output_file, 'w') as file:
    
    file.write(f"#rem:{count}\n")
    file.write("def getPointlist():\n")
    file.write("\t pointList=[")
    
    #for feature in layer.getFeatures(): 
    for feature in layer.selectedFeatures():
        attribute_value = feature[attribute_name]

        if counter==count-1:
            attribute_value2=attribute_value[:-1]
        else:
            attribute_value2=attribute_value
        file.write(f"\t{attribute_value2}\n")
        counter=counter+1
    
    file.write(f"\t ]\n")
    file.write(f"\t return pointList\n")
   

print(f" {count} attribute values written to {output_file}")