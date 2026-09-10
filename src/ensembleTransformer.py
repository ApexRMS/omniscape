## omniscape

# Ensemble Connectivity transformer
#
# Combines the normalized current maps of two or more omniscape Scenarios
# (typically one per species) into a single ensemble connectivity surface.
#
# The Scenarios to combine are supplied as dependencies of this Scenario. Each
# one declares which ensemble member it represents - typically a species - in
# its own 'Ensemble Membership' datasheet, and this Scenario's 'Ensemble
# Weights' says what each member is worth. Weighting by member rather than by
# Scenario is what lets one Scenario count for different amounts in different
# ensembles. Anything left unset weighs 1.0.
#
# Classification is deliberately not done here. Run 'Categorize Connectivity
# Output' after this transformer to slice the ensemble into connectivity
# categories; it picks the ensemble raster up automatically and can classify it
# either against fixed values or by quantile.

import pysyncrosim as ps
import pandas as pd
import os
import rasterio
import numpy as np
import sys

from helperFunctions import (safe_progress_bar, resolve_list_option,
                             resolve_boolean_option)
from ensembleFunctions import (NODATA_VALUE, nodata_mask, validate_same_grid,
                               standardize_min_max, focal_statistic, combine_layers,
                               read_scenario_member, resolve_ensemble_weights,
                               safe_update_run_log)


# Set up -----------------------------------------------------------------------

safe_progress_bar(message="Setting up Scenario", report_type="message")

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

outputEnsemblePath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatialEnsemble")
if os.path.exists(outputEnsemblePath) == False:
    os.makedirs(outputEnsemblePath)


# Load input and settings from SyncroSim Library --------------------------------

ensembleOptions = myScenario.datasheets(name = "omniscape_ensembleOptions")
ensembleWeights = myScenario.datasheets(name = "omniscape_ensembleWeights")

# The Project's member vocabulary, used to report members by name rather than
# by the ID the datasheets store
ensembleMembers = myProject.datasheets(name = "omniscape_ensembleMembers", include_key = True)
memberNames = {}
if not ensembleMembers.empty:
    memberNames = {int(r.ensembleMembersId): str(r.Name)
                   for r in ensembleMembers.itertuples()}


# Resolve options, tolerating both list IDs and display names --------------------

COMBINATION_NAMES = {0: "Weighted Mean", 1: "Weighted Sum", 2: "Maximum", 3: "Minimum"}
FOCAL_NAMES = {0: "Mean", 1: "Sum", 2: "Max", 3: "Min"}

if ensembleOptions.empty:
    combinationMethod = "Weighted Mean"
    standardizeInputs = True
    useFocalWindow = False
    focalRadius = None
    focalFunction = "Mean"
else:
    combinationMethod = resolve_list_option(ensembleOptions.combinationFunction.item(), COMBINATION_NAMES, "Weighted Mean")
    standardizeInputs = resolve_boolean_option(ensembleOptions.standardizeInputs.item(), True)
    useFocalWindow = resolve_boolean_option(ensembleOptions.useFocalWindow.item(), False)
    focalRadius = ensembleOptions.focalRadius.item()
    focalFunction = resolve_list_option(ensembleOptions.focalFunction.item(), FOCAL_NAMES, "Mean")

if useFocalWindow:
    if focalRadius != focalRadius or focalRadius is None:
        sys.exit("'Use focal window' is enabled, therefore 'Focal radius' is required.")
    focalRadius = int(focalRadius)
    if focalRadius < 1:
        sys.exit("'Focal radius' must be at least 1 pixel.")


# Identify the Scenarios to combine from the dependencies ------------------------

dependencyTable = myParentScenario.dependencies

# Fix the dependency order once, here, and use this same table everywhere below,
# so that the run log reports the dependencies in the order they were combined
dependencyTable = dependencyTable.sort_values(by = "Priority").reset_index(drop = True)

if len(dependencyTable) < 2:
    sys.exit(
        "The Ensemble Connectivity transformer requires at least 2 Scenario "
        "dependencies, but " + repr(len(dependencyTable)) + " were found. Add "
        "each omniscape Scenario to combine (typically one per species) as a "
        "dependency of this Scenario.")

safe_update_run_log("Ensemble combination method: " + combinationMethod
                    + ("; inputs standardized to 0-1" if standardizeInputs else
                       "; inputs NOT standardized")
                    + ((("; focal " + focalFunction + ", radius "
                         + repr(focalRadius) + " px") if useFocalWindow else "")) + ".")


# Load each dependency's normalized current raster -------------------------------

safe_progress_bar(message="Loading dependency Scenarios", report_type="message")

allScenarios = myProject.scenarios(optional = True)

layerList = []
maskList = []
memberList = []
referenceRaster = None

for depRow in dependencyTable.itertuples():
    depId = int(depRow.Id)
    depName = str(depRow.Name)

    # Resolve a parent Scenario to its most recent result
    depTable = allScenarios[allScenarios.ScenarioId == depId]
    if "Yes" in np.unique(depTable.IsResult):
        depScenario = myLibrary.scenarios(depId)
    else:
        depResults = allScenarios[allScenarios.ParentId == depId]
        if depResults.empty:
            sys.exit("No results were found for dependency Scenario '"
                     + depName + "' (ID " + repr(depId) + ").")
        depScenario = myLibrary.scenarios(int(max(depResults.ScenarioId)))

    depOutput = depScenario.datasheets(name = "omniscape_outputSpatial", show_full_paths = True)

    if depOutput.empty or depOutput.normalizedCumCurrmap[0] != depOutput.normalizedCumCurrmap[0]:
        sys.exit("A 'Normalized current' raster is required for dependency "
                 "Scenario '" + depName + "' (ID " + repr(depId) + ").")

    depRaster = rasterio.open(depOutput.normalizedCumCurrmap[0])

    if referenceRaster is None:
        referenceRaster = depRaster
    else:
        validate_same_grid(referenceRaster, depRaster, "Normalized current ('" + depName + "')")

    depData = depRaster.read(1).astype(float)
    depMask = nodata_mask(depRaster, depData)

    if standardizeInputs:
        depData = standardize_min_max(depData, depMask)

    # Appended together with the layer it belongs to, so a Scenario's membership
    # cannot drift onto a different Scenario's raster.
    #
    # Membership is read from the Scenario the user actually listed as a
    # dependency, not from the result the raster came out of. A result carries a
    # copy of its inputs as they stood when it was produced, so reading it there
    # would silently use a stale value whenever membership changed after that
    # dependency was last run - and would read nothing at all from a result
    # produced before this datasheet existed. Where the dependency IS a result,
    # this resolves to that same result, which is what was asked for.
    layerList.append(depData)
    maskList.append(depMask)
    memberList.append(read_scenario_member(myLibrary.scenarios(depId), depName, memberNames))

weightsList, weightsMessage = resolve_ensemble_weights(
    dependencyTable, memberList, ensembleWeights, memberNames)
safe_update_run_log(weightsMessage)


# Combine into the ensemble -------------------------------------------------------

safe_progress_bar(message="Combining Scenarios into ensemble", report_type="message")

ensembleData, ensembleMask = combine_layers(
    np.stack(layerList), np.stack(maskList), weightsList, combinationMethod)

if useFocalWindow:
    safe_progress_bar(message="Applying focal window", report_type="message")
    ensembleData = focal_statistic(ensembleData, ensembleMask, focalRadius, focalFunction)


# Save spatial output --------------------------------------------------------------

outMeta = referenceRaster.meta.copy()
outMeta.update(count = 1, dtype = "float32", nodata = NODATA_VALUE)

ensembleOut = np.where(ensembleMask, NODATA_VALUE, ensembleData).astype("float32")
ensemblePath = os.path.join(outputEnsemblePath, "ensemble_connectivity.tif")
with rasterio.open(ensemblePath, mode = "w", **outMeta) as outputRaster:
    outputRaster.write(ensembleOut, 1)

outputSpatialEnsemble = myScenario.datasheets(name = "omniscape_outputSpatialEnsemble")
outputSpatialEnsemble.ensembleRaster = pd.Series(ensemblePath)
myParentScenario.save_datasheet(name = "omniscape_outputSpatialEnsemble", data = outputSpatialEnsemble)

validCount = int((~ensembleMask).sum())
safe_update_run_log(
    "Ensemble connectivity written (" + repr(validCount) + " valid pixels). Add "
    "'Categorize Connectivity Output' to the pipeline after this stage to classify it.")
