## omniscape

# Set up -----------------------------------------------------------------------

from osgeo import gdal
import pysyncrosim as ps
import pandas as pd
import sys
import os
import rasterio
import numpy as np

ps.environment.progress_bar(message="Setting up Scenario", report_type="message")

e = ps.environment._environment()
wrkDir = e.data_directory.item()

if os.path.exists(wrkDir) == False:
    os.mkdir(wrkDir)

myLibrary = ps.Library()
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)
myScenarioParentID = int(myScenario.parent_id)
myParentScenario = myLibrary.scenarios(sid = myScenarioParentID)

dataPath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID)) 



# Load input and settings from SyncroSim Library ------------------------------- 

requiredData = myScenario.datasheets(name = "omniscape_Required")
requiredDataValidation = myScenario.datasheets(name = "omniscape_Required", show_full_paths=True)
generalOptions = myScenario.datasheets(name = "omniscape_GeneralOptions")
resistanceOptions = myScenario.datasheets(name = "omniscape_ResistanceOptions")
reclassTable = myScenario.datasheets(name = "omniscape_ReclassTable")
condition1 = myScenario.datasheets(name = "omniscape_Condition1")
condition2 = myScenario.datasheets(name = "omniscape_Condition2")
conditionalOptions = myScenario.datasheets(name = "omniscape_ConditionalOptions")
futureConditions = myScenario.datasheets(name = "omniscape_FutureConditions")
outputOptions = myScenario.datasheets(name = "omniscape_OutputOptions")
multiprocessing = myScenario.datasheets(name = "core_Multiprocessing")
juliaConfig = myScenario.datasheets(name = "omniscape_juliaConfiguration")



# If not provided, set default values  -----------------------------------------

if generalOptions.sourceFromResistance.item() == "Yes":
   requiredData.sourceFile = pd.Series("None")

if generalOptions.blockSize.empty:
    generalOptions.blockSize = pd.Series(1)

if generalOptions.sourceFromResistance.item() != generalOptions.sourceFromResistance.item():
    generalOptions.sourceFromResistance = pd.Series("No")

if generalOptions.resistanceIsConductance.item() != generalOptions.resistanceIsConductance.item():
    generalOptions.resistanceIsConductance = pd.Series("No")

if generalOptions.rCutoff.item() != generalOptions.rCutoff.item():
    generalOptions.rCutoff = pd.Series("Inf")

if generalOptions.buffer.item() != generalOptions.buffer.item():
    generalOptions.buffer = pd.Series(0)

if generalOptions.sourceThreshold.item() != generalOptions.sourceThreshold.item():
    generalOptions.sourceThreshold = pd.Series(0)

if generalOptions.calcNormalizedCurrent.item() != generalOptions.calcNormalizedCurrent.item():
    generalOptions.calcNormalizedCurrent = pd.Series("No")

if generalOptions.calcFlowPotential.item() != generalOptions.calcFlowPotential.item():
    generalOptions.calcFlowPotential = pd.Series("No")

if generalOptions.allowDifferentProjections.item() != generalOptions.allowDifferentProjections.item():
    generalOptions.allowDifferentProjections = pd.Series("No")

if generalOptions.connectFourNeighborsOnly.item() != generalOptions.connectFourNeighborsOnly.item():
    generalOptions.connectFourNeighborsOnly = pd.Series("No")

if generalOptions.solver.item() != generalOptions.solver.item():
    generalOptions.solver = pd.Series("cg+amg")

if resistanceOptions.reclassifyResistance.empty:
    resistanceOptions.reclassifyResistance = pd.Series("No")

if resistanceOptions.reclassifyResistance.item() == "No":
    resistanceOptions.reclassTable = pd.Series("None")

if resistanceOptions.writeReclassifiedResistance.item() != resistanceOptions.writeReclassifiedResistance.item():
    resistanceOptions.writeReclassifiedResistance = pd.Series("Yes")

if conditionalOptions.conditional.empty:
    conditionalOptions.conditional = pd.Series("No")

if conditionalOptions.nConditions.item() != conditionalOptions.nConditions.item():
    conditionalOptions.nConditions = pd.Series(1)

if conditionalOptions.conditional.item() == "No":
    condition1.condition1File = pd.Series("None")
    condition1.condition1Lower = pd.Series("NaN")
    condition1.condition1Upper = pd.Series("NaN")
    condition2.condition2File = pd.Series("None")
    condition2.condition2Lower = pd.Series("NaN")
    condition2.condition2Upper = pd.Series("NaN")

if condition1.comparison1.item() != condition1.comparison1.item():
    condition1.comparison1 = pd.Series("within")

if condition2.comparison2.item() != condition2.comparison2.item():
    condition2.comparison2 = pd.Series("within")

if futureConditions.compareToFuture.empty:
    futureConditions.compareToFuture = pd.Series("none")

if futureConditions.compareToFuture.item() == "none":
    futureConditions.condition1FutureFile = pd.Series("None")
    futureConditions.condition2FutureFile = pd.Series("None")

if outputOptions.writeRawCurrmap.empty:
    outputOptions.writeRawCurrmap = pd.Series("Yes")

if outputOptions.maskNodata.item() != outputOptions.maskNodata.item():
    outputOptions.maskNodata = pd.Series("Yes")

if outputOptions.writeAsTif.item() != outputOptions.writeAsTif.item():
    outputOptions.writeAsTif = pd.Series("Yes")




# Validation -------------------------------------------------------------------

if juliaConfig.juliaPath.empty:
    sys.exit("A julia executable is required.")

if not os.path.isfile(juliaConfig.juliaPath.item()):
    sys.exit("The path to the julia executable is not valid or does not exist.")

if ' ' in juliaConfig.juliaPath.item():
    sys.exit("The path to the julia executable may not contains spaces.")

if not 'julia.exe' in juliaConfig.juliaPath.item():
    sys.exit("The path to the julia executable must contain the 'julia.exe' file.")

if multiprocessing.EnableMultiprocessing.item() == "Yes":
    if multiprocessing.MaximumJobs.empty or multiprocessing.MaximumJobs.item() < 1:
        sys.exit("'Maximum Jobs' must be at least 1 when multiprocessing is enabled.")

if requiredData.resistanceFile.item() != requiredData.resistanceFile.item():
    sys.exit("'Resistance file' is required.")

resistanceLayer = rasterio.open(requiredDataValidation.resistanceFile[0])
dataRaster = resistanceLayer.read()
unique, counts = np.unique(dataRaster, return_counts = True)
unique = pd.DataFrame(unique)
if (unique[0] <= 0).values.any():
    sys.exit("'Resistance file' may not contain 0 or negative values.")

if requiredData.radius[0] != requiredData.radius[0]:
    sys.exit("'Radius' is required.")

if generalOptions.sourceFromResistance.item() == "No" and requiredData.sourceFile.item() == "None":
    sys.exit("'Source from resistance' was set to 'No', therefore 'Source file' is required.")

if generalOptions.sourceFromResistance.item() == "No" and requiredDataValidation.sourceFile.item() == requiredDataValidation.sourceFile.item():
    resistanceLayer = rasterio.open(requiredDataValidation.resistanceFile.item())
    sourceLayer = rasterio.open(requiredDataValidation.sourceFile.item())
    if resistanceLayer.crs != sourceLayer.crs:
        sys.exit("'Resistance file' and 'Source file' must have the same Coordinate Reference System.")
    if resistanceLayer.bounds != sourceLayer.bounds:
        sys.exit("'Resistance file' and 'Source file' must have the same raster extent.")

if not resistanceOptions.empty:
    if resistanceOptions.reclassifyResistance.item() == "Yes":
        if reclassTable.empty:
            sys.exit("'Reclassify resistance' was set to 'Yes', therefore 'Reclass Table' is required.")
        if reclassTable['landCover'].isnull().values.any():
            sys.exit("'Reclass Table' has NaN values for 'Land cover class'.")
        if reclassTable['resistanceValue'].isnull().values.any():
            sys.exit("'Reclass Table' has NaN values for 'Resistance value'. If necessary, NaN values should be specified as -9999.")

if not conditionalOptions.empty:
    if conditionalOptions.conditional.item() == "Yes" and conditionalOptions.nConditions.item() == "1" and condition1.condition1File.item() != condition1.condition1File.item():
        sys.exit("'Conditional' was set to 'Yes' and 'Number of conditions' was set to 1, therefore 'Condition 1 file' is required.")
    if conditionalOptions.conditional.item() == "Yes" and conditionalOptions.nConditions.item() == 2 and (condition1.condition1File.item() != condition1.condition1File.item() or condition2.condition2File.item() != condition2.condition2File.item()):
        sys.exit("'Conditional' was set to 'Yes' and 'Number of conditions' was set to 2, therefore 'Condition 1 file' and 'Condition 2 file' are required.")
    if condition1.comparison1.item() == "within" and (condition1.condition1Lower.item() != condition1.condition1Lower.item() or condition1.condition1Upper.item() != condition1.condition1Upper.item()):
        sys.exit("'Comparison 1' was set to 'within', therefore 'Condition 1 lower' and 'Condition 1 upper' are required.")
    if condition2.comparison2.item() == "within" and (condition2.condition2Lower.item() != condition2.condition2Lower.item() or condition2.condition2Upper.item() != condition2.condition2Upper.item()):
        sys.exit("'Comparison 2' was set to 'within', therefore 'Condition 2 lower' and 'Condition 2 upper' are required.")

if not futureConditions.empty:
    if futureConditions.compareToFuture.item() == "1" and futureConditions.condition1FutureFile.item() != futureConditions.condition1FutureFile.item():
        sys.exit("'Compare to future' was set to 1, therefore 'Condition 1 future file' is required.")
    if futureConditions.compareToFuture.item() == "2" and futureConditions.condition2FutureFile.item() != futureConditions.condition2FutureFile.item():
        sys.exit("'Compare to future' was set to 2, therefore 'Condition 2 future file' is required.")
    if futureConditions.compareToFuture.item() == "both" and (futureConditions.condition1FutureFile.item() != futureConditions.condition1FutureFile.item() or futureConditions.condition2FutureFile.item() != futureConditions.condition2FutureFile.item()):
        sys.exit("'Compare to future' was set to 'both', therefore 'Condition 1 future file' and 'Condition 2 future file' are required.")



# Change "Yes" and "No" to "true" and "false" ----------------------------------

generalOptions = generalOptions.replace({'Yes': 'true', 'No': 'false'})
resistanceOptions = resistanceOptions.replace({'Yes': 'true', 'No': 'false'})
conditionalOptions = conditionalOptions.replace({'Yes': 'true', 'No': 'false'})
outputOptions = outputOptions.replace({'Yes': 'true', 'No': 'false'})
multiprocessing = multiprocessing.replace({'Yes': 'true', 'No': 'false'})



# Prepare reclass file ---------------------------------------------------------

ps.environment.progress_bar(message="Preparing for Omniscape run", report_type="message")

if not reclassTable.empty and (reclassTable != "None").values.any():
    reclassTablePath = os.path.join(dataPath, "omniscape_ResistanceOptions")
    if os.path.exists(reclassTablePath) == False:
        os.mkdir(reclassTablePath)
    #reclassTable.resistanceValue = reclassTable.resistanceValue.astype(str)
    reclassTable.loc[reclassTable["resistanceValue"] == -9999, "resistanceValue"] = "missing"
    with open(os.path.join(reclassTablePath, "reclass_table.txt"), "w") as f:
        file = reclassTable.to_string(header=False, index=False)
        f.write(file)
else:
    reclassTablePath = "None"



# Prepare configuration file (.ini) --------------------------------------------

file = open(os.path.join(dataPath, "omniscape_Required", "config.ini"), "w")
file.write(
    "[Required]" + "\n"
    "resistance_file = " + os.path.join(dataPath, "omniscape_Required", requiredData.resistanceFile.item()) + "\n"
    "radius = " + repr(requiredData.radius.item()) + "\n"
    "project_name = " + os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial") + "\n"
    "source_file = " + requiredData.sourceFile.item() + "\n"
    "\n"
    "[General Options]" + "\n"
    "block_size = " + repr(generalOptions.blockSize.item()) + "\n"
    "source_from_resistance = " + generalOptions.sourceFromResistance.item() + "\n"
    "resistance_is_conductance = " + generalOptions.resistanceIsConductance.item() + "\n"
    "r_cutoff = " + repr(generalOptions.rCutoff.item()) + "\n"
    "buffer = " + repr(generalOptions.buffer.item()) + "\n"
    "source_threshold = " + repr(generalOptions.sourceThreshold.item()) + "\n"
    "calc_normalized_current = " + generalOptions.calcNormalizedCurrent.item() + "\n"
    "calc_flow_potential = " + generalOptions.calcFlowPotential.item() + "\n"
    "allow_different_projections = " + generalOptions.allowDifferentProjections.item() + "\n"
    "connect_four_neighbors_only = " + generalOptions.connectFourNeighborsOnly.item() + "\n"
    "solver = " + generalOptions.solver.item() + "\n"
    "\n"
    "[Resistance Reclassification]" + "\n"
    "reclassify_resistance = " + resistanceOptions.reclassifyResistance.item() + "\n"
    "reclass_table = " + os.path.join(reclassTablePath, "reclass_table.txt") + "\n"
    "write_reclassified_resistance = " + resistanceOptions.writeReclassifiedResistance.item() + "\n"
    "\n"
    "[Conditional Connectivity]" + "\n"
    "conditional = " + conditionalOptions.conditional.item() + "\n"
    "n_conditions = " + repr(conditionalOptions.nConditions.item()) + "\n"
    "condition1_file = " + condition1.condition1File.item() + "\n"
    "comparison1 = " + condition1.comparison1.item() + "\n"
    "condition1_lower = " + condition1.condition1Lower.item() + "\n"
    "condition1_upper = " + condition1.condition1Upper.item() + "\n"
    "condition2_file = " + condition2.condition2File.item() + "\n"
    "comparison2 = " + condition2.comparison2.item() + "\n"
    "condition2_lower = " + condition2.condition2Lower.item() + "\n"
    "condition2_upper = " + condition2.condition2Upper.item() + "\n"
    "compare_to_future = " + futureConditions.compareToFuture.item() + "\n"
    "condition1_future_file = " + futureConditions.condition1FutureFile.item() + "\n"
    "condition2_future_file = " + futureConditions.condition2FutureFile.item() + "\n"
    "\n"
    "[Output Options]" + "\n"
    "write_raw_currmap = " + outputOptions.writeRawCurrmap.item() + "\n"
    "mask_nodata = " + outputOptions.maskNodata.item() + "\n"
    "write_as_tif = " + outputOptions.writeAsTif.item() + "\n"
    "\n"
    "[Multiprocessing]" + "\n"
    "parallelize = " + multiprocessing.EnableMultiprocessing.item() + "\n"
    "parallel_batch_size = " + repr(multiprocessing.MaximumJobs.item()) + "\n"
)
file.close()



# Prepare julia script ---------------------------------------------------------

configName = "config.ini"

file = open(os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl"), "w")
file.write(
    "cd(raw\"" + os.path.join(dataPath, "omniscape_Required") + "\")" + "\n"
    "\n"
    "using Pkg; Pkg.add(name=\"GDAL\"); Pkg.add(name=\"Omniscape\")" + "\n"
    "using Omniscape" + "\n"
    "run_omniscape(\"" + configName + "\")"
)
file.close()



# Run julia script with system call -------------------------------------------------------------

ps.environment.progress_bar(message="Running Omniscape", report_type="message")

jlExe = juliaConfig.juliaPath.item()
runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")

if ' ' in dataPath:
    sys.exit("Due to julia requirements, the path to the SyncroSim Library may not contain any spaces.")

# Add thread specification if multiprocessing is enabled
if multiprocessing.EnableMultiprocessing.item() == "true":
    numThreads = int(multiprocessing.MaximumJobs.item())
    runOmniscape = f"{jlExe} -t {numThreads} {runFile}"
else:
    runOmniscape = f"{jlExe} {runFile}"

os.system(runOmniscape)



# Create output datasheets ----------------------------------------------------------------------

myOutput = myScenario.datasheets(name = "omniscape_outputSpatial")

if outputOptions.writeRawCurrmap.item() == "true":
    myOutput.cumCurrmap = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "cum_currmap.tif"))

if generalOptions.calcFlowPotential.item() == "true":
    myOutput.flowPotential = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "flow_potential.tif"))

if generalOptions.calcNormalizedCurrent.item() == "true":
    myOutput.normalizedCumCurrmap = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "normalized_cum_currmap.tif"))

if (os.path.isfile(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "classified_resistance.tif"))) & (resistanceOptions.writeReclassifiedResistance.item() == "true"):
    myOutput.classifiedResistance = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "classified_resistance.tif"))
else:
    myOutput.classifiedResistance = pd.Series(requiredDataValidation.resistanceFile.item())



# Save outputs to SyncroSim ---------------------------------------------------------------------

myParentScenario.save_datasheet(name = "omniscape_outputSpatial", data = myOutput)


