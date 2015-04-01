# gerrymadering

A Study of Gerrymandering

# Data sources and files

## Compactness

### Shapefiles and compactness calculations

- source: https://www.census.gov/geo/maps-data/data/tiger-line.html
- notebook: python/Get_compactness_score.ipynb
- data: data/compactness113_byGEO.json

## Popular vote

### 2012

- source: manually downloaded and edited "Candidate Details" HTML table from http://www.thegreenpapers.com/G12/HouseVoteByParty.phtml
- data (raw): data/113_2012_house_popular_vote.csv
- notebook: python/save_113_2012_house_popular_vote.ipynb
- data (processed): data/113_2012_house_popular_vote.json

### 2014

#### Primary data (includes 3rd party votes)
- source: downloaded Sheet1 from https://docs.google.com/spreadsheet/ccc?key=0AjYj9mXElO_QdHVsbnNNdXRoaUE5QThHclNWaTgzb2c&usp=drive_web#gid=0 as CSV
  - This was found in the external links section of http://en.wikipedia.org/wiki/United_States_House_of_Representatives_elections,_2014
- data (raw): data/114_2014_house_popular_vote.csv
- notebook: python/save_114_2014_house_popular_vote.ipynb
- data (processed): data/114_2014_house_popular_vote.json

#### Secondary data (includes census data, not analyzed)
- source: downloaded first sheet from https://docs.google.com/spreadsheets/d/1lGZQi9AxPHjE0RllvhNEBHFvrJGlnmC43AXnR8dwHMc/
- data: data/114_2014_house_election_2010census.csv

## Redistricting

- source: http://redistricting.lls.edu/who-partyfed.php
- notebook: python/get_redistricting_authorities.ipynb
- data: data/redistricting_2010.json

## PVI

### Districts

- source: http://cookpolitical.com/file/2013-04-49.pdf
- data: data/pvi_district.csv

### States

- source: http://cookpolitical.com/file/filename.pdf
- data: data/pvi_state.csv

## DIME

- source: http://data.stanford.edu/dime
- notebook: python/DIME_load_save.ipynb
- note: you will have to download and run this notebook to get the dataset

# Utilities

## Convert between state names, abbreviations, and FIPS numbers.

- data/st2Fips.json: State (2 letter abbreviation) to FIPS numbers
- data/state2Fips.json: State (full name) to FIPS numbers
- data/st2State.json: State (2 letter abbreviation) to full name
- data/state2St.json: State (full name) to 2 letter abbreviation

## Functions

python/congress_tools.py: collection of functions used in processing data

# Analysis

## Expected number of representatives

notebook: python/nExpectedReps_compactSummary_114.ipynb

## Compactness and Redistricting

notebook: python/Redistricting_and_compactness.ipynb

## Compactness and District PVI

notebook: python/Cooks_data_and_compactness_analysis.ipynb

## Gerrymandering Score

notebook: python/gerry_score.ipynb

## GM Score and DIME

notebook: python/DIME_analysis.ipynb

# Visualization

## Exporting for d3.js

notebook: python/export_data_for_visualization.ipynb

## Circles with district shapes

notebook: python/Maps_and_viz.ipynb
