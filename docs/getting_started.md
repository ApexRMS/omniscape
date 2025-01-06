---
layout: default
title: Getting started
permalink: /getting_started
---

# Getting started with **omniscape**

---
## ⚠️ **Notice to Users**

The **Getting Started** documentation and associated **Tutorials** for this SyncroSim package currently reflects information for **SyncroSim version 2**. We are in the process of updating these pages to ensure compatibility with **SyncroSim version 3**.
In the meantime, please note that some instructions, references, and/or images may not fully align with the latest version of SyncroSim. We appreciate your patience as we work to provide updated resources.

---

To get started working with **omniscape** SyncroSim, begin by:
<br>
1. <a href="#installing-syncrosim-and-julia">Installing SyncroSim and Julia</a>
2. <a href="#installing-the-omniscape-syncrosim-package">Installing the **omniscape** SyncroSim package</a>

<br>

## **Installing SyncroSim and Julia**

Running **omniscape** SyncroSim requires that SyncroSim and Julia be installed on your computer. 
1. Download SyncroSim version 3.0 or greater [here](https://syncrosim.com/download/){:target="_blank"} and follow the installation prompts. 
2. Download Julia version 1.9 or greater [here](https://julialang.org/downloads/){:target="_blank"} and follow the installation prompts.

<br>

## **Installing the omniscape SyncroSim package**

1\. Open SyncroSim Desktop.

2\. Select **Files > Packages**.

<img align="center" style="padding: 13px" width="300" src="assets/images/screenshot1.png">

3\. The *Packages* window will open, listing all the SyncroSim packages installed in your computer. To install a new package from the Package Server, click **Install**.

<img align="center" style="padding: 13px" width="600" src="assets/images/screenshot2.png">

4\. A new window will open listing the packages available for install from the Package Server. To install **omniscape** SyncroSim, mark the checkbox beside the package name and click **OK**. 

<img align="center" style="padding: 13px" width="600" src="assets/images/screenshot3-new.png">

5\.	The **omniscape** SyncroSim package uses Conda to manage the package dependencies. Upon installing the package, you will be prompted to install Conda if it is not already installed in your computer. Then, you will be prompted to create or update the Conda environment for **omniscape** SyncroSim. Click **Yes**.  

<img align="center" style="padding: 13px" width="600" src="assets/images/screenshot4.png">

6\.	Return to the *Packages* window, **omniscape** will now be listed along with the other installed packages, and the Conda checkbox will be marked.

<img align="center" style="padding: 13px" width="600" src="assets/images/screenshot5-new.png">

## **Next steps**

Once the requirements have been installed, the following tutorials will cover the basics of the **omniscape** and **omniscapeImpact** SyncroSim packages for connectivity analyses:
1. <a href="./tutorials/omniscape">Reproducing the Omniscape.jl example with **omniscape** SyncroSim</a>
2. <a href="./tutorials/omniscapeImpact">Measuring the impact of connectivity change with <b>omniscapeImpact</b></a>

<img align="center" style="padding: 13px" width="800" src="assets/images/screenshot6.png">

<br><br><br>