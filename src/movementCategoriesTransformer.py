## omniscape

# Connectivity Categories transformer 

# Set up -----------------------------------------------------------------------

from osgeo import gdal
import pysyncrosim as ps
import pandas as pd
import os
import sys
import rasterio
import numpy as np
from helperFunctions import safe_progress_bar

safe_progress_bar(message = "Setting up Scenario", report_type = "message")

e = ps.environment._environment()
wrkDir = e.data_directory.item()

myLibrary = ps.Library()
myProject = myLibrary.projects(pid = 1)
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)

# Handle parent scenario (may be NaN in partial libraries during tiling)
if pd.isna(myScenario.parent_id):
    myParentScenario = myScenario  # Use self as parent if no parent exists
else:
    myScenarioParentID = int(myScenario.parent_id)
    myParentScenario = myLibrary.scenarios(sid = myScenarioParentID)

dataPath = os.path.join(e.data_directory.item(), "Scenario-" + repr(myScenarioID))

# Create directory, if applicable
outputMovementPath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatialMovement")
if os.path.exists(outputMovementPath) == False:
    os.makedirs(outputMovementPath)


# Load input and output datasheet from the SyncroSim Library ------------------- 

myInput = myScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
movementTypeClasses = myProject.datasheets(name = "omniscape_movementTypes", include_key = True)
reclassificationThresholds = myScenario.datasheets(name = "omniscape_reclassificationThresholds")
TabularReclassification = myScenario.datasheets(name = "omniscape_outputTabularReclassification")
myOutput = myScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)



# Validation -------------------------------------------------------------------

if myInput.normalizedCumCurrmap.item() != myInput.normalizedCumCurrmap.item():
    sys.exit("'Categorize Connectivity Output' was added to the pipeline. Therefore, a 'Normalized current' raster is required.")

if movementTypeClasses.empty:
    sys.exit("'Categorize Connectivity Output' was added to the pipeline. Therefore, the 'Connectivity Categories' datasheet is required.")

if reclassificationThresholds.empty:
    sys.exit("'Categorize Connectivity Output' was added to the pipeline. Therefore, the 'Category Thresholds' datasheet is required.")



# Categorize connectivity output ----------------------------------------------------------------

safe_progress_bar(message = "Categorizing connectivity output", report_type = "message")

normCurr = rasterio.open(myInput.normalizedCumCurrmap.item())
data = normCurr.read()
reclassRaster = data.copy()

for i in reclassificationThresholds.index:
    reclassRaster[np.where((data >= reclassificationThresholds['minValue'][i]) & (data < reclassificationThresholds['maxValue'][i]))] = movementTypeClasses['classID'][i]

outMeta = normCurr.meta
outMeta.update(dtype = "int16")
with rasterio.open(
    os.path.join(outputMovementPath, "connectivity_categories.tif"), 
    mode="w", **outMeta) as outputRaster:
    outputRaster.write(reclassRaster)

myOutput.movementTypes = pd.Series(os.path.join(outputMovementPath, "connectivity_categories.tif"))

unique, counts = np.unique(reclassRaster, return_counts = True)
unique = pd.DataFrame(unique)
unique[0] = unique[0].astype(int)
freq = pd.DataFrame(counts)
uniqueFreq = pd.concat([unique, freq], axis = 1, ignore_index = True)
movementFreq = uniqueFreq[(uniqueFreq[0].isin(movementTypeClasses.classID))]
movementFreq = movementFreq.rename(columns = {0: "classID", 1:"freq"})

movementTypesFreq = pd.merge(right = movementFreq, left = movementTypeClasses)

percentCover = movementTypesFreq.freq/movementTypesFreq.freq.sum()
amountArea = (movementTypesFreq.freq * normCurr.res[1] * normCurr.res[1])/10000

tabularMovementTypes = pd.concat([movementTypesFreq.movementTypesId, amountArea, percentCover], axis = 1, ignore_index = True)
myTabularOutput = tabularMovementTypes.rename(columns = {0: "movementTypesID", 1:"amountArea", 2:"percentCover"})

myParentScenario.save_datasheet(name = "omniscape_outputTabularReclassification", data = myTabularOutput)



# Save outputs to SyncroSim ---------------------------------------------------------------------

myParentScenario.save_datasheet(name = "omniscape_outputSpatialMovement", data = myOutput)


