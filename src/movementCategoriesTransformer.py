## omniscape

# Connectivity Categories transformer
#
# Slices a connectivity surface into the Project's connectivity categories.
#
# The surface is whichever of these the Scenario has produced:
#   - the ensemble raster, when 'Ensemble Connectivity' ran earlier in the
#     pipeline, or
#   - the normalized current map from 'Omniscape' otherwise.
#
# Categories are defined either against fixed raster values (the default) or by
# quantile, in which case the break values are computed from this run's own
# distribution. Quantiles suit a surface whose range is not known ahead of time
# - a combined ensemble, for instance, whose values depend on how many
# Scenarios went into it and how they were weighted.

# Set up -----------------------------------------------------------------------

import pysyncrosim as ps
import pandas as pd
import os
import sys
import rasterio

from helperFunctions import (NODATA_VALUE, safe_progress_bar, safe_update_run_log,
                             nodata_mask, resolve_list_option, validate_threshold_bands,
                             classify_by_quantiles, classify_by_values)

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

omniscapeOutput = myScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)
ensembleOutput = myScenario.datasheets(name = "omniscape_outputSpatialEnsemble", show_full_paths = True)
movementTypeClasses = myProject.datasheets(name = "omniscape_movementTypes", include_key = True)
reclassificationOptions = myScenario.datasheets(name = "omniscape_reclassificationOptions")
reclassificationThresholds = myScenario.datasheets(name = "omniscape_reclassificationThresholds")
myOutput = myScenario.datasheets(name = "omniscape_outputSpatialMovement", show_full_paths = True)


# Choose which surface to categorize -------------------------------------------

def firstValue(datasheet, column):
    """Return a populated value from a single-row output datasheet, or None."""
    if datasheet.empty or column not in datasheet.columns:
        return None
    value = datasheet[column].iloc[0]
    if value != value or value is None:      # NaN
        return None
    return value


ensemblePath = firstValue(ensembleOutput, "ensembleRaster")
normalizedPath = firstValue(omniscapeOutput, "normalizedCumCurrmap")

# The ensemble wins when both exist: an ensemble Scenario is asking for its
# combined surface to be classified, not whatever single-Scenario output may
# also be sitting in the same Scenario
if ensemblePath is not None:
    inputPath = ensemblePath
    inputLabel = "Ensemble connectivity"
elif normalizedPath is not None:
    inputPath = normalizedPath
    inputLabel = "Normalized current"
else:
    sys.exit(
        "'Categorize Connectivity Output' was added to the pipeline, so it needs a surface "
        "to categorize. Run 'Omniscape' earlier in the pipeline to produce a 'Normalized "
        "current' raster, or 'Ensemble Connectivity' to produce an ensemble raster.")


# Resolve options ---------------------------------------------------------------

THRESHOLD_TYPE_NAMES = {0: "Value", 1: "Quantile"}

if reclassificationOptions.empty:
    thresholdType = "Value"
else:
    thresholdType = resolve_list_option(
        reclassificationOptions.thresholdType.item(), THRESHOLD_TYPE_NAMES, "Value")


# Validation -------------------------------------------------------------------

if movementTypeClasses.empty:
    sys.exit("'Categorize Connectivity Output' was added to the pipeline. Therefore, the 'Connectivity Categories' datasheet is required.")

if reclassificationThresholds.empty:
    sys.exit("'Categorize Connectivity Output' was added to the pipeline. Therefore, the 'Category Thresholds' datasheet is required.")

# Quantiles are proportions of the distribution, so they are bounded by 0 and 1;
# raw values are bounded only by the raster itself
if thresholdType == "Quantile":
    validate_threshold_bands(reclassificationThresholds, "minValue", "maxValue",
                             "Category Thresholds", lower_limit = 0.0, upper_limit = 1.0)
else:
    validate_threshold_bands(reclassificationThresholds, "minValue", "maxValue",
                             "Category Thresholds")

# Attach classIDs to the threshold rows through the Project's category
# vocabulary. Matching by name rather than by row position matters: the two
# datasheets are independently ordered, so pairing them positionally silently
# assigns the wrong category whenever their orders differ.
thresholdTable = reclassificationThresholds.merge(
    movementTypeClasses[["movementTypesId", "Name", "classID"]],
    left_on = "movementType", right_on = "Name", how = "left", validate = "many_to_one")

if thresholdTable.classID.isna().any():
    unknown = thresholdTable[thresholdTable.classID.isna()].movementType.tolist()
    sys.exit("'Category Thresholds' references unknown connectivity categories: "
             + ", ".join(repr(u) for u in unknown) + ".")


# Categorize connectivity output ----------------------------------------------------------------

safe_progress_bar(message = "Categorizing connectivity output", report_type = "message")

safe_update_run_log(
    "Categorizing '" + inputLabel + "' (" + os.path.basename(str(inputPath)) + ") "
    "using " + thresholdType.lower() + " thresholds.")

inputRaster = rasterio.open(inputPath)
data = inputRaster.read(1).astype(float)
mask = nodata_mask(inputRaster, data)

if thresholdType == "Quantile":
    # Rename so the shared classifier sees the quantile columns it expects
    quantileTable = thresholdTable.rename(
        columns = {"minValue": "minQuantile", "maxValue": "maxQuantile"})
    reclassRaster, breaksTable = classify_by_quantiles(data, mask, quantileTable)
else:
    reclassRaster, breaksTable = classify_by_values(data, mask, thresholdTable)

outMeta = inputRaster.meta.copy()
outMeta.update(count = 1, dtype = "int16", nodata = NODATA_VALUE)

categoriesPath = os.path.join(outputMovementPath, "connectivity_categories.tif")
with rasterio.open(categoriesPath, mode = "w", **outMeta) as outputRaster:
    outputRaster.write(reclassRaster.astype("int16"), 1)

myOutput.movementTypes = pd.Series(categoriesPath)


# Save tabular output ----------------------------------------------------------------

# Per category: the break values actually used (computed from this run's
# distribution in quantile mode, taken straight from the datasheet otherwise),
# plus the resulting area and share of valid pixels
pixelArea = abs(inputRaster.res[0] * inputRaster.res[1])
validCount = int((~mask).sum())

summaryRows = []

for breakRow in breaksTable.itertuples():
    classCount = int((reclassRaster == breakRow.classID).sum())
    movementTypesId = int(movementTypeClasses.movementTypesId[
        movementTypeClasses.classID == breakRow.classID].iloc[0])
    summaryRows.append({
        "movementTypesID": movementTypesId,
        "minBreakValue": float(breakRow.minBreakValue),
        "maxBreakValue": float(breakRow.maxBreakValue),
        "amountArea": (classCount * pixelArea) / 10000,
        "percentCover": (classCount / validCount) if validCount > 0 else 0.0})

myTabularOutput = pd.DataFrame(summaryRows)

# The category reference must stay an integer. Assigning a whole row positionally
# (df.loc[n] = [...]) coerces every column in that row to float, which submits the
# category as "16.0" rather than "16"; SyncroSim then cannot match it against the
# Connectivity Categories list and rejects the save.
if not myTabularOutput.empty:
    myTabularOutput["movementTypesID"] = myTabularOutput["movementTypesID"].astype("int64")

myParentScenario.save_datasheet(name = "omniscape_outputTabularReclassification", data = myTabularOutput)


# Save outputs to SyncroSim ---------------------------------------------------------------------

myParentScenario.save_datasheet(name = "omniscape_outputSpatialMovement", data = myOutput)
