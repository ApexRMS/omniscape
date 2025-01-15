---
layout: default
title: Home
description: "SyncroSim package for calculating omnidirectional landscape connectivity"
permalink: /
---

# **omniscape** & **omniscapeImpact** SyncroSim

<br>

<img align="right" style="padding: 13px" width="180" src="assets/images/logo/omniscape-sticker.png">
[![GitHub release](https://img.shields.io/github/v/release/ApexRMS/omniscape.svg?style=for-the-badge&color=d68a06)](https://GitHub.com/ApexRMS/omniscape/releases/) <a href="https://github.com/ApexRMS/omniscape"><img align="middle" style="padding: 1px" width="30" src="assets/images/logo/github-trans2.png"></a> omniscape
<br>
[![GitHub release](https://img.shields.io/github/v/release/ApexRMS/omniscapeImpact.svg?style=for-the-badge&color=d68a06)](https://GitHub.com/ApexRMS/omniscapeImpact/releases/)
<a href="https://github.com/ApexRMS/omniscape"><img align="middle" style="padding: 1px" width="30" src="assets/images/logo/github-trans2.png"></a> omniscapeImpact
<br>

## Omnidirectional landscape connectivity based on circuit theory

**omniscape** SyncroSim is an open-source [SyncroSim](https://syncrosim.com/){:target="_blank"} package for running [Omniscape.jl](https://docs.circuitscape.org/Omniscape.jl/stable/){:target="_blank"} and calculating omnidirectional landscape connectivity. 

**omniscape** SyncroSim allows users to run the latest Omniscape.jl code without ever having to interact with Julia directly. Rather, through SyncroSim, users can run an Omniscape analysis using either SyncroSim Studio or through scripts written in R (using [rsyncrosim](https://syncrosim.com/r-package/){:target="_blank"}) or Python (using [pysyncrosim](https://pysyncrosim.readthedocs.io/en/latest/){:target="_blank"}). For details on model parameters and configurations, see the [Omniscape.jl documentation](https://docs.circuitscape.org/Omniscape.jl/stable/usage/#Settings-and-Options){:target="_blank"}.

**omniscapeImpact** is [package](https://docs.syncrosim.com/how_to_guides/package_overview.html){:target="_blank"} that compares the connectivity outputs of two **omniscape** analyses. For more information, see the [omniscapeImpact tutorial](https://apexrms.github.io/omniscape/tutorials/omniscapeImpact).

<br> 

## Requirements

The latest version of **omniscape** SyncroSim has two requirements:
* SyncroSim [version 3.0](https://syncrosim.com/download/){:target="_blank"} or greater
* Julia [version 1.9](https://julialang.org/downloads/){:target="_blank"} or greater

Instructions for installing the above requirements are provided on the [Getting Started](https://apexrms.github.io/omniscape/getting_started.html) page.

<br>

## Tutorials

For tutorials covering the basics of **omniscape** and **omniscapeImpact** SyncroSim packages, see:
* <a href="./tutorials/omniscape">Reproducing the Omniscape.jl example with **omniscape** SyncroSim</a>
* <a href="./tutorials/omniscapeImpact">Measuring the impact of connectivity change with <b>omniscapeImpact</b></a>

<br>

## Key Links

Browse source code for **omniscape** at
[http://github.com/ApexRMS/omniscape/](http://github.com/ApexRMS/omniscape/){:target="_blank"}. <br>
Browse source code for **omniscapeImpact** at
[http://github.com/ApexRMS/omniscapeImpact/](http://github.com/ApexRMS/omniscapeImpact/){:target="_blank"}. <br>
Report a bug or contribute an idea at
[http://github.com/ApexRMS/omniscape/issues](http://github.com/ApexRMS/omniscape/issues){:target="_blank"}. <br>
Omniscape.jl documentation at [https://docs.circuitscape.org/Omniscape.jl/stable/](https://docs.circuitscape.org/Omniscape.jl/stable/){:target="_blank"}. <br>

<br>

## Contributors

Carina Rauen Firkowski (Author, maintainer) <a href="https://orcid.org/0000-0003-0540-9529"><img align="middle" style="padding: 0.5px" width="17" src="assets/images/ORCID.png"></a>
<br>
Bronwyn Rayfield (Author) <a href="https://orcid.org/0000-0003-1768-1300"><img align="middle" style="padding: 0.5px" width="17" src="assets/images/ORCID.png"></a>
<br>
Katie Birchard (Author) <a href="https://orcid.org/0009-0003-7519-4751"><img align="middle" style="padding: 0.5px" width="17" src="assets/images/ORCID.png"></a>
<br>
Marie-Josée Fortin (Author) <a href="https://orcid.org/0000-0002-9935-1366"><img align="middle" style="padding: 0.5px" width="17" src="assets/images/ORCID.png"></a>
<br>
Colin Daniel (Author) <a href="https://orcid.org/0000-0001-7367-2041"><img align="middle" style="padding: 0.5px" width="17" src="assets/images/ORCID.png"></a>

<br>

## Citation

<a href="https://doi.org/10.5281/zenodo.10818776" target="_blank"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10818776.svg" alt="DOI"></a>

Firkowski, C., Rayfield, B., Birchard, K., Fortin, M.-J., & Daniel, C. (2023). omniscape SyncroSim: Omni-directional habitat connectivity based on circuit theory (1.0.1). ApexRMS. [https://doi.org/10.5281/zenodo.10818776](https://doi.org/10.5281/zenodo.10818776).
