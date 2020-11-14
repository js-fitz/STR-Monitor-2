# STR Tracker

### An online dashboard for tracking short-term rentals in Boston MA

#### Using M-L clustering to reverse-anonymize high-traffic Airbnb building location anonymity. 


##### Data Sources: InsideAirbnb, City of Boston, Alliance of Downtown Civic Organizations (ADCO)

- The entire site — https://air-monitor.onrender.com/ — is comprised of static html pages each containing the relevant JS and CSS. Each page is generated procedurally using the code contained in this repo. 

- Core code is in the `mods` directory.
    - `cluster.py` is for prepping the data, including applying the SKLEARN DBSCAN algorithm to establish geographical groups of listings
    - `map.py` generates the Folium map objects and web pages for all pages on the final site
    
- `driver.ipynb` provides a one-click solution to re-generate the site when new data becomes available using both `mods`.

- `airbnb_tracker` contains all the code hosted for the live site. This is an exact replica of the data currently live on:
   - https://air-monitor.onrender.com/
   
