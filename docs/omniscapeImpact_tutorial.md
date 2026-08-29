---
layout: default
title: OmniscapeImpact
permalink: tutorials/omniscapeImpact
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
<br>

## **Measuring the impact of connectivity change with omniscapeImpact**

This tutorial guides you through using the **omniscapeImpact** package, which extends **omniscape SyncroSim**. It covers the following steps:

1. <A href="#step-1">Installing the omniscapeImpact package</A>
2. <A href="#step-2">Creating and configuring an omniscapeImpact SyncroSim Library</A>
3. <A href="#step-3">Visualizing and comparing scenario results</A>

<br>

### **Requirements**

Before you begin, make sure that the **omniscape** SyncroSim package version 2.7.0 is installed, along with SyncroSim version 3.1.0 or greater. For more information, see <A href="https://apexrms.github.io/omniscape/getting_started#installing-the-omniscape-syncrosim-package">Installing the **omniscape** SyncroSim package</A>.

> **Note:** **omniscapeImpact** shares its Conda environment with **omniscape**, so **omniscape** must be installed and its Conda environment created first.

The Baseline and Alternative Scenarios being compared must be directly comparable. **omniscapeImpact** will stop with an error if their output rasters differ in extent, resolution or projection, or if they disagree about which pixels hold valid data. If the two Scenarios were classified using different *Category Thresholds*, the connectivity category outputs are skipped and only the *Normalized current* comparison is produced.

<br>

<p id="step-1"> <h3><b>Step 1. Installing the omniscapeImpact package</b></h3> </p>

1\.	Open SyncroSim Desktop.

2\.	Select **File > Packages**.

<img align="center" style="padding: 13px" width="250" src="./images/screenshot43.png">

3\.	The *Packages* window will open, listing all the SyncroSim packages installed in your computer. To install a new package from the Package Server, click **Install**.

<img align="center" style="padding: 13px" width="550" src="./images/screenshot44-new.png">

4\.	A new window will open listing the packages available for install from the Package Server. To install **omniscapeImpact**, mark the checkbox beside the package name and click **OK**. 

<img align="center" style="padding: 13px" width="550" src="./images/screenshot45-new.png">

5\.	The **omniscapeImpact** package uses Conda to manage the package dependencies. Upon installing the package, you will be prompted to create or update the Conda environment for **omniscapeImpact**. Click **Yes**.

<img align="center" style="padding: 13px" width="550" src="./images/screenshot46.png">

6\.	Return to the *Packages* window. **omniscapeImpact** will now be listed along with the other installed packages, and the Conda checkbox will be marked.

<img align="center" style="padding: 13px" width="550" src="./images/screenshot47-new.png">

<br>

<p id="step-2"> <h3><b>Step 2. Creating and configuring an omniscapeImpact SyncroSim Library</b></h3> </p>

1\.	Open SyncroSim Desktop.

2\.	To create a new library, select **File > New**.

<img align="center" style="padding: 13px" width="250" src="./images/screenshot48.png">

<div class=indentation>
  a.	From the list of packages, select <b>omniscape</b>.
  <br><br>
  b.	Select the <b>Omniscape Impact</b> template library. If desired, you may edit the <i>File name</i>, and change the <i>Folder</i> by clicking on the <i>Browse</i> button. Click <b>OK</b>.
</div>

<img align="center" style="padding: 13px" width="550" src="./images/screenshot49-new.png">

<br>

A new library will be created based on the selected template, and SyncroSim will automatically open and display it in the *Library Explorer* window.

<img align="center" style="padding: 13px" width="375" src="./images/screenshot50.png">

1\.	Note that the library contains two folders: *omniscape* and *omniscapeImpact*.

2\.	Expand the **omniscape** folder and note that it contains three scenarios.

<img align="center" style="padding: 13px" width="375" src="./images/screenshot51.png">

<div class=indentation>
The first two scenarios were covered in the tutorial <A href="omniscape">Reproducing the Omniscape.jl example with <b>omniscape</b> SyncroSim</A>. 
<br><br>
The additional scenario, <i>Decreased resistance</i>, represents the case where resistance has been decreased by a similar magnitude as in the <i>Increased resistance</i> scenario.
</div>

3\.	Next, expand the **omniscapeImpact** folder and note that it contains two scenarios:

<div class=indentation>
  a.	<i>Impact of increased resistance</i> – compares the <i>Reference resistance</i> and <i>Increased resistance</i> scenarios.
  <br><br>
  b.	<i>Impact of decreased resistance</i> – compares the <i>Reference resistance</i> and <i>Decreased resistance</i> scenarios.
</div>

<img align="center" style="padding: 13px" width="400" src="./images/screenshot52.png">

4\.	Double-click on *Impact of increased resistance* to open the scenario properties.

5\.	Under the *General* tab, navigate to the **Pipeline** datasheet.

<div class=indentation>
Note that it lists one pipeline stage, <i>Connectivity Impact Assessment</i>.
</div>

<img align="center" style="padding: 13px" width="550" src="./images/screenshot53.png">

6\.	Close the scenario properties and, in the *Library Explorer* window, expand the **Impact of increased resistance** scenario to expose its **Dependencies** folder.

<div class=indentation>
The two scenarios to be compared are supplied as <i>dependencies</i> of the impact scenario, added by dragging each scenario onto the <i>Dependencies</i> folder. <b>The order matters</b>:
<div class=indentation>
  i.	The <b>first</b> dependency is the <i>Baseline</i> – the reference connectivity state from which changes will be measured. For this example, it is the <i>Reference resistance</i> scenario.
  <br><br>
  ii.	The <b>second</b> dependency is the <i>Alternative</i> – the changed connectivity state. For this example, it is the <i>Increased resistance</i> scenario.
</div>
<br>
To swap the Baseline and Alternative, reorder the dependencies. The run log for each run records which scenario was used as which. Adding a scenario (rather than one of its results) as a dependency automatically uses that scenario's most recent valid results.
</div>
7\.	Collapse the scenario.

<br>

<p id="step-3"> <h3><b>Step 3. Visualizing and comparing scenario results</b></h3> </p>

The Omniscape Impact template library already contains the results for all its scenarios. In SyncroSim, the results for a scenario are organized into a Results folder, nested within its parent scenario. 

1\.	In the Library Explorer window, click on the arrow beside the **Impact of increased resistance** scenario to expose the *Results* folder; repeat the same action to expose the results scenario. 

<img align="center" style="padding: 13px" width="400" src="./images/screenshot55.png">

2\.	Double-click on the results scenario to open its properties.

3\.	Navigate to the **Add-on** tab and expand the **Results** node. 

<img align="center" style="padding: 13px" width="450" src="./images/screenshot56.png">

<br>

The **omniscapeImpact** package generates spatial and tabular outputs.

4\.	Under the *Spatial* node are the following outputs:

<div class=indentation>
  a.	<i>Overall</i> – represents per pixels change in normalized current flow or connectivity category. 
  <br><br>
  b.	<i>Per Category</i> – represents per pixels loss, gain and no change for each connectivity category.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot57.png">

5\.	Under the *Tabular* node are the following outputs:

<div class=indentation>
  a.	<i>Differences Summary</i> – represents the change in area and proportion between the baseline and alternative scenarios for each connectivity category.
  <br><br>
  b.	<i>Transitions Summary</i> – represents the change in area and proportion between the baseline and alternative scenarios for all possible transitions between connectivity categories.
  <br><br>
  c.	<i>Jaccard Dissimilarity</i> – represents the dissimilarity between the baseline and alternative scenarios for each connectivity category. For each connectivity category, the Jaccard Dissimilarity is calculated as 1 minus the ratio between the number of shared pixels across scenarios and the total number of pixels across scenarios.
</div>

<img align="center" style="padding: 13px" width="600" src="./images/screenshot58-new.png">

6\.	Close the scenario properties and collapse the results and scenario folder.

<div class=indentation>
You will now visualize the tabular outputs of <b>omniscapeImpact</b>
</div>

7\.	In the Library Explorer window, select the scenarios **Impact of increased resistance** and **Impact of decreased resistance**, right-click, and select **Add to Results** from the context menu. 

<img align="center" style="padding: 13px" width="400" src="./images/screenshot59-new.png">

8\.	Navigate to the **Charts** tab and double-click to view the **Area difference** chart. 

<img align="center" style="padding: 13px" width="600" src="./images/screenshot60-new.png">

<div class=indentation>
Note that the two scenarios had opposite effects on connectivity. 
<br><br>
The Increased resistance scenario led to an increase in area for <i>Impeded</i> and <i>Channelized</i> and a decrease in area for <i>Diffuse</i> and <i>Intensified</i>, relative to the Reference scenario. 
<br><br>
The Decreased resistance scenario led to a decrease in area for <i>Impeded</i> and <i>Channelized</i> and an increase in area for <i>Diffuse</i> and <i>Intensified</i>, relative to the Reference scenario. 
<br><br>
Note also that the magnitude of change in area was larger under the Decreased resistance scenario.
</div>

9\.	Close the *Area difference* chart and open the **Jaccard dissimilarity** chart.

<img align="center" style="padding: 13px" width="600" src="./images/screenshot61-new.png">

<div class=indentation>
Note that a similar pattern is reflected here, with <i>Impact of decreased resistance</i> showing greater dissimilarity to the baseline scenario compared to <i>Impact of increased resistance</i>.
</div>

<br>

Next, for a visual confirmation of the quantitative changes summarized by area and the Jaccard dissimilarity, you will inspect the spatial outputs of **omniscapeImpact**.

10\.	Navigate to the **Maps** tab and double-click to open the **Normalized current difference** and **Cross-category difference** maps.

<img align="center" style="padding: 13px" width="650" src="./images/screenshot62.png">

<div class=indentation>
The <i>Normalized current difference</i> summarizes continuous change in current between the baseline and alternative scenarios.
<br><br>
In turn, the <i>Cross-category difference</i> map highlights pixels where the change in current represented a change in connectivity category. For example, a change from <i>Impeded</i> to <i>Diffuse</i> would represent a <i>1 category gain</i>, while a change from <i>Channelized</i> to <i>Diffuse</i> would represent a <i>2 category loss</i>.
</div>

11\.	Close the *Normalized current difference* and open the **Per-category difference** map.

<img align="center" style="padding: 13px" width="650" src="./images/screenshot63.png">

<div class=indentation>
The <i>Per-category difference</i> map represents per pixel losses and gains for each category. Together with the <i>Cross-category difference</i> map, it can be used to identify transitions between connectivity categories.
<br><br>
Both maps also highlight that a decrease in resistance had a stronger effect on connectivity than an increase.
</div>

<br><br><br>
