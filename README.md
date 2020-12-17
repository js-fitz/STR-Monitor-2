[under construction...]

# STR Tracker
## Dashboard for track Short-Term Rentals

The [dashboard](https://air-monitor.onrender.com/) is compiled by an html page generator function, with cross-links embedded for navigation. JS and CSS are generated dynamically for the <head> of head page.

## Key features:
- Accounts for known groups of alias accounts used by the same company
- Identifies clusters of listings that may belong to the same building (partially reversing anonymized locations)
- Tracks buildings over time to calculate changes in the number of listings (and to maintain static cluster index names)
- Generates files for an easy-to-host static dashboard website

## Detailed methodology:
### `cluster.py`:


- **Verify license statuses**
  1. Parse license numbers / claimed exemptions from Airbnb data and from City Registry data
  2. For each Airbnb listing, get the license details (status, address, etc) from the Registry if found. else listing license is fabricated or missing (none claimed) 
  3. For each matched license, make sure the number of listings/beds doesn't exceed the number allowed according to the license type

- **Find probable groups of listings belonging to one building (for each host)** (double DBSCAN)
    1. For corporations known to use multiple Airbnb accounts, group all of their listings into a "Corporate Group" account
    2. For each host (or corp. group), DBSCAN to find geographically dense clusters of listings that share the same license (**mininum 5 listings per group**)
        - This is possible because in Boston at least, many hosts re-use license numbers across multiple listings
        - Groups of fewer than five: no clusters for outliers
    4. DBSCAN again to check for overlaps between clusters — i.e. multiple licenses in one building. Join the any overlapping clusters for each host.
    5. If previous data exists, check for any similar clusters (close & similar size) in case this is an update to a previous cluster. This way there is also a static cluster naming system.
    3. Check for sub-clusters within each cluster (boolean optional, currently disabled) — multiple smaller clusters detected within, sometimes buildings within a complex


- **Calculate statistics on probable building locations**
    1. Average lat/long of all listings in cluster for the cluster centroid
    2. Calc. various stats on dispersal of listings in cluster (max distance from center, centroid confidence)
    3. Reverse-geocode centroid location to get full "best-guess" address 
        - For listings with a license match in the city registry, use the address from there (no 'best guess')

##### Data Sources: InsideAirbnb, City of Boston, Alliance of Downtown Civic Organizations (ADCO)

- 

- `mods` contains the functions 
    - `mods/cluster.py` preps the data, including applying the DBSCAN algo. to establish probable building clusters
        - also gets a lot of stats about the data...
    - `mods/map.py` creates all the Folium map objects, sidebar elements, and saves final html pages
    
- `driver.ipynb`: an example of how to re-generate the site when new data becomes available (uses `cluster` and `map`)

- `airbnb_tracker` contains the files used in the live site:
   - https://air-monitor.onrender.com/
   
