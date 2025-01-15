---
layout: default
title: Tutorials
permalink: tutorials/omniscape
---

<style>
  .indentation {
    margin-left: 1rem;
    margin-top: 1rem; 
    margin-bottom: 1rem; 
  }
</style>

---
## ⚠️ **Notice to Users**

The **Getting Started** documentation and associated **Tutorials** for this SyncroSim package currently reflects information for **SyncroSim version 2**. We are in the process of updating these pages to ensure compatibility with **SyncroSim version 3**.
In the meantime, please note that some instructions, references, and/or images may not fully align with the latest version of SyncroSim. We appreciate your patience as we work to provide updated resources.

---

## **Reproducing the Omniscape.jl example with omniscape SyncroSim**

This tutorial provides an overview of working with **omniscape** SyncroSim in the Windows user interface. It covers the following steps:

1. <A href="#step-1">Creating and configuring an omniscape SyncroSim Library</A>
2. <A href="#step-2">Visualizing scenario results</A>
3. <A href="#step-3">Creating, editing, and running a new scenario</A>
4. <A href="#step-4">Comparing results across scenarios</A>

<br>

<p id="step-1"> <h3><b>Step 1. Creating and configuring an omniscape SyncroSim Library</b></h3> </p>

In SyncroSim, a library is a file with extension .ssim that stores all the model’s inputs and outputs in a format specific to a given package. To create a new library:

1\.	Open SyncroSim Desktop.

2\.	Select **File > New**.

<img align="center" style="padding: 13px" width="250" src="./images/screenshot6.png">

<div class=indentation>
a. From the list of packages, select <b>omniscape</b>. Three template library options will be available: Empty Library, Omniscape Example, and Omniscape Impact.
<br><br>
b. Select the <b>Omniscape Example</b> template library. If desired, you may edit the <i>File name</i>, and change the <i>Folder</i> by clicking on the <i>Browse</i> button. Click <b>OK</b>.
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot7-new.png">

<br>

A new library has been created based on the selected template. SyncroSim will automatically open and display it in the Library Explorer window.

3\.	Double-click on the library name, **Omniscape Example**, to open the library properties window. You may also right-click on the library name and select **Properties** from the context menu.

<img align="center" style="padding: 13px" width="400" src="./images/screenshot8.png">

4\.	The *Summary* datasheet contains the metadata for the library.

<img align="center" style="padding: 13px" width="600" src="./images/screenshot9-new.png">

5\.	Navigate to the **Julia Configuration tab**.

<div class=indentation> 
The path to the <b>Julia executable</b> file must be specified to run <b>omniscape</b> SyncroSim. To do so, click on the folder icon and navigate to where Julia is installed in your computer. Its default location is <b>C:\Users\[User_Name]\AppData\Local\Programs\Julia-[version]\bin\julia.exe</b>. 
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot10-new.png">

> **Note:** The AppData folder is sometimes hidden. To see it, in File Explorer, select View > Show > Hidden items.

6\.	Next, navigate to the **Options** tab.

<div class=indentation> 
In the <i>General</i> datasheet, mark the checkbox for <b>Use conda</b>.
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot11.png">

7\.	Close the library properties window.

<br>

Next, you will review the inputs of the *Reference resistance* scenario. In SyncroSim, each scenario contains the model inputs and outputs associated with a model run.

1\.	In the Library Explorer window, select the pre-configured scenario **Reference resistance** and double-click it to open its properties. You may also right-click on the scenario name and select **Properties** from the context menu.

<img align="center" style="padding: 13px" width="450" src="./images/screenshot12.png">

2\.	Navigate to the **Pipeline** datasheet.

<img align="center" style="padding: 13px" width="450" src="./images/screenshot13.png">

<div class=indentation>
a. Under the <i>Stage</i> column, note that two pipeline stages are set in the following order:
<br>
<div class=indentation>
    i.	<i>Omniscape</i> – runs Omniscape.jl 
    <br><br>
    ii.	<i>Categorize Connectivity Output</i> – classifies the continuous output from Omniscape.jl into connectivity categories based on a set of threshold values
    <br>
  </div>
  Each pipeline stage calls on a transformer (<i>i.e.</i>, script) which takes the inputs from SyncroSim, runs a model, and returns the results to SyncroSim.
</div>

<br>

The *Omniscape* pipeline stage replicates the exact structure and order of parameters as Omniscape.jl with inputs organized in two tabs: *Required* and *Optional*. 

3\.	Navigate to the **Required** tab, which contains the following inputs:

<div class=indentation>
  a. <i>Resistance file</i> – a raster file of land cover or resistance classes. For this example, the pre-loaded raster corresponds to the 2016 National Land Cover Dataset for central Maryland.
  <br><br>
  b. <i>Radius</i> – sets the radius of the moving window. This example uses a radius of 100 pixels.
  <br><br>
  c. <i>Source file</i> – a raster file indicating which pixels correspond to sources. In this example, the sources are set through a different method, described in the next step.
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot14.png">

4\.	Navigate to the **Optional** tab.

<div class=indentation>
  a.	Under the <i>General Options</i> datasheet, note that <i>Source from resistance</i> is set to <i>Yes</i>, enforcing that the sources be calculated from the resistance layer based on a threshold of 1 as defined by <i>R cutoff</i>.
</div>
  
<img align="center" style="padding: 13px" width="500" src="./images/screenshot15.png">

<div class=indentation>
  b. Navigate to the <b>Resistance Reclassification</b> node and review the following inputs:
  <br>
  <div class=indentation>
    i.	<i>Options > Reclassify resistance</i> – determines whether the <i>Resistance file</i> should be reclassified. For this example, it is set to <i>Yes</i> since the <i>Resistance file</i> provided in step 11.a corresponded to a raster of land cover classes.
    <br><br>
    ii.	<i>Options > Write reclassified resistance</i> – determines whether the reclassified resistance raster should be saved and written to file. 
    </div>
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot16.png">

<div class=indentation>
<div class=indentation>
  iii.	<i>Reclass table</i> – a reclassification matrix used to translate land cover classes into resistance values.
  </div>
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot17.png">

<br>

The *Categorize Connectivity Output* pipeline stage is an exclusive feature of the **omniscape** SyncroSim package. It allows for seamless post-processing of the continuous output of Omniscape into discrete connectivity categories based on user-defined connectivity categories, a common step in the Omniscape workflow.

5\.	In the Library Explorer, double-click on **Definitions** to open the project properties. You may also right-click on the project name and select **Properties** from the context menu. 

<img align="center" style="padding: 13px" width="375" src="./images/screenshot18.png">

6\.	Under the *Summary* datasheet, the *Description* field highlights that the connectivity categories and thresholds used in the template library were derived from Cameron *et al.* (2022, Conservation Science and Practice).

7\.	Navigate to the **Connectivity Categories** tab.

<div class=indentation>
  a.	Note that four connectivity categories have been defined, each associated with an ID value.
  <br><br>
  b.	Click on the <b>Category ID</b> column to sort categories in ascending order, where <i>Impeded</i> represents areas with the least amount of flow, and <i>Channelized</i> represents areas with the greatest amount to flow.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot19.png">

8\.	Return to the scenario properties window, navigate to the **Advanced** tab.

<img align="center" style="padding: 13px" width="600" src="./images/screenshot20.png">

<div class=indentation>
Note that each connectivity category is associated with a minimum and maximum value, defining the range of normalized current flow that will be reclassified into each connectivity category.
</div>

9\.	Close the scenario properties window.

<br>

<p id="step-2"> <h3><b>Step 2. Visualizing scenario results</b></h3> </p>

The Omniscape Example template library already contains the results for the *Reference resistance* scenario. In SyncroSim, the results for a scenario are organized into a *Results* folder, nested within its parent scenario. 

1\.	In the Library Explorer window, click on the arrow beside the *Reference resistance* scenario to expose the *Results* folder; repeat the same action to expose the results scenario. 

<img align="center" style="padding: 13px" width="375" src="./images/screenshot21.png">

2\.	Double-click on the results scenario to open its properties.

<div class=indentation>
a. Click through the <b>Required</b>, <b>Optional</b> and <b>Advanced</b> tabs and note that the results scenario is a copy of the parent scenario’s inputs, which are greyed out.
</div> 
  
<img align="center" style="padding: 13px" width="400" src="./images/screenshot22.png">

<div class=indentation>
b. Navigate to the <b>Results</b> tab, which lists the spatial and tabular outputs. Both pipeline stages have spatial outputs but only the second pipeline stage has tabular outputs.
<br>
<div class=indentation>
i. Under the <b>Spatial</b> node, the outputs from the first and second transformers are organized under the <i>Omniscape Outputs</i> and <i>Connectivity Categories</i> datasheets, respectively.
<br><br>
You can export any spatial output by clicking on the <i>Export</i> button.
</div>
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot23.png">

<div class=indentation>
<div class=indentation>
ii.	Under the <b>Tabular</b> node, the output of the second transformer is saved to the <i>Connectivity Categories Summary</i> datasheet.
<br><br>
You can export the tabular output by right-clicking on the data and selecting <i>Export All</i> from the context menu.
</div>
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot24-new.png">

3\.	Close the results scenario properties.

<div class=indentation>
Using the SyncroSim built-in tools, you will now visualize the outputs of the first transformer.
</div>

4\.	In the Library Explorer window, right-click on the **Reference resistance** scenario and select **Add to Results** from the context menu. 

<img align="center" style="padding: 13px" width="375" src="./images/screenshot25.png">

5\.	Navigate to the **Maps** tab and double-click on the first pre-configured map, **Cumulative current flow**. 

<img align="center" style="padding: 13px" width="350" src="./images/screenshot26.png">

<div class=indentation>
The cumulative current flow represents the total current flowing through the landscape. To inspect the map, consider the following:
</div>

<div class=indentation>
  a.	<i>Map legend</i> – displayed along the left-hand side of the window. It can be edited by double-clicking it.
  <br><br>
  b.	<i>Toolbar</i> – displayed along the top of the window. Includes zoom, pan and per pixel information tooltip.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot27.png">

6\.	Close the map window and view the two following maps.

<div class=indentation>
  a.	<b>Flow potential</b> represents current flow under the null condition of resistance set to 1 for the entire landscape. 
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot28.png">

<div class=indentation>
  b.	<b>Normalized current flow</b> is calculated as flow potential divided by cumulative current, and therefore represents where there is more or less current than expected under null resistance conditions.
</div>

<img align="center" style="padding: 13px" width="500" src="./images/screenshot29.png">

<br>

Next, you will visualize the outputs of the second transformer, which takes the *Normalized current flow* map and reclassifies it into a discrete map, based on the threshold values reviewed in steps 13-17.

7\.	View the **Connectivity categories** map.

8\.	Keep the map open for comparison and from the Library Explorer window, navigate to the **Charts** tab and view the **Area** chart. 

<img align="center" style="padding: 13px" width="600" src="./images/screenshot30-new.png">

<div class=indentation>
Note that it summarized the amount of area per connectivity category.
Alternatively, a <b>Proportion</b> summary is also available.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot31-new.png">

9\.	Close all plot windows.

<br>

<p id="step-3"> <h3><b>Step 3. Creating, editing, and running a new scenario</b></h3> </p>

Next, you will learn how to create a scenario and run it to generate results. This scenario will differ from the *Reference resistance* by a ten-fold increase in resistance for all non-forest (i.e., non-source) pixels. 

1\.	Right-click on the existing scenario and select **Copy** from the context menu. 

<img align="center" style="padding: 13px" width="400" src="./images/screenshot32.png">

<div class=indentation>
Then, right-click anywhere inside the Library Explorer window and select <b>Paste</b>
</div>

<img align="center" style="padding: 13px" width="400" src="./images/screenshot33.png">

2\.	Double-click on the new scenario to open its properties.

<div class=indentation>
  a.	Change the <b>Name</b> to <i>Increased resistance</i>. 
  <br><br>
  b.	Change the <b>Description</b> to <i>Resistance values for non-forest land cover types were increased by one order of magnitude. All other configuration options for Omniscape are equal to those implemented in the Omniscape.jl example</i>.
</div>

<img align="center" style="padding: 13px" width="550" src="./images/screenshot34.png">

3\.	Navigate to the **Optional** tab and under the **Resistance Reclassification** node, open the **Reclass table** datasheet.

<div class=indentation>
  a.	Under the <b>Resistance value</b> column, increase values by one order of magnitude, except for <i>Land cover class</i> 41, 42, and 43, which represent forest classes.
</div>

<img align="center" style="padding: 13px" width="550" src="./images/screenshot35.png">

4\.	Close the scenario properties window and save the changes to the library.

<br>

Next, you will run the scenario. For this example, the run should take approximately 10 minutes with multiprocessing enabled across 5 cores.

5\.	To enable multiprocessing, click on the **Multiprocessing** button along the SyncroSim toolbar and adjust the number of **Multiprocessing jobs** to 5.

<img align="center" style="padding: 13px" width="175" src="./images/screenshot36.png">

6\.	In the Library Explorer window, right-click on the **Increased resistance** scenario and select **Run** for the context menu.

<img align="center" style="padding: 13px" width="400" src="./images/screenshot37.png">

<div class=indentation>
  a.	The <i>Run Monitor</i> window will open, informing that the model is <i>Running</i>.
</div>

<img align="center" style="padding: 13px" width="450" src="./images/screenshot38.png">

<div class=indentation>
  b.	Along the bottom-right of the window, a progress bar will provide further details.
  <br><br>
  c.	First, SyncroSim calls the first transformer, <i>Omniscape</i>. 
<div class=indentation>
    i.	<i>Setting up Scenario</i> – the transformer takes the inputs and pre-processes them into the format required by Julia. 
    <br><br>
    ii.	<i>Running Omniscape</i> – the transformer calls Julia to run the analysis. Once the analysis is complete, the transformer retrieves and saves the outputs back to SyncroSim. 
    </div>
  d.	Then, SyncroSim call the second transformer, <i>Categorize Connectivity Output</i>. 
  <div class=indentation>
    i.	<i>Setting up Scenario</i> – the transformer takes an output from the first transformer along with the connectivity categories and their threshold values.
    <br><br>
    ii.	<i>Categorizing connectivity output</i> – the transformer reclassifies the continuous output and returns it back to SyncroSim.
    </div>
</div>

7\.	When the run is complete, the *Status* will be updated to *Done*. You can inspect the **Run Log**, which returns the total run time for the scenario. 

<img align="center" style="padding: 13px" width="500" src="./images/screenshot39.png">

<br>

<p id="step-4"> <h3><b>Step 4. Comparing results across scenarios</b></h3> </p>

With two successful scenario runs, you will now compare their results.

1\.	Ensure that both scenarios are added to the results. This is noted by a red check mark beside the scenario symbol and a bolded scenario name. 

<img align="center" style="padding: 13px" width="325" src="./images/screenshot40.png">

<div class=indentation>
If required, right-click on the scenario(s) and select <b>Add to Results</b> from the context menu.
</div>

2\.	First, view the **Area** chart.

<img align="center" style="padding: 13px" width="600" src="./images/screenshot41-new.png">

<div class=indentation>
Note that the increase in resistance resulted in a small increase in the amount of <i>Impeded</i> and <i>Channelized</i> areas, and a decrease in the amount of <i>Diffuse</i> and <i>Intensified</i> areas.  
</div>

3\.	Next, view the **Connectivity categories** map. 

<div class=indentation>
Zoom in and pan through the map and try to identify where across the landscape those changes in connectivity occurred.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot42.png">

<div class=indentation>
Visually identifying areas of change between scenarios may not be straightforward. For more quantitative tools to compare changes in connectivity, see the next tutorial <A href="omniscapeImpact">Measuring the impact of connectivity change with <b>omniscapeImpact</b></A>.
</div>

<br><br><br>