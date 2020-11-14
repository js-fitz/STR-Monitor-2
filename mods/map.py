# This mod contains functions generate maps & HTML pages, using prepped data


import os
import math
import re
import json
import time
import config
import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
# to normalize & standardize opacity
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# manual date info (ISD last updated) 
config.ISD_updated = 'Aug. 2020'


old = pd.read_csv('data_raw/listings_old.csv') # scrape before current data
last_scraped = pd.to_datetime(old.last_scraped)
config.last_scraped = last_scraped.min().strftime('%b. %Y')
print('Previous scrape detected:', config.last_scraped)
latest = pd.read_csv('data_raw/listings_old.csv')
latest_scraped = pd.to_datetime(latest.last_scraped)
config.latest_scraped = latest_scraped.max().strftime('%b. %Y')
print('Latest scrape detected:', config.latest_scraped)


# turn sub-clustering on or off
def sub_mode(active=False):
    config.sub_mode = active

## UNIVERSAL ELEMENT COLORS [BY LICENSE STATUS]
status_color_dict = {
    'No license claimed':  '#FF9800', # also used to set the order of the legend (by key)
    'Exempt (not verified)': '#9E9E9E',
    'Exempt (not verified): Hospital Stays': '#9E9E9E',
    'Exempt (not verified): Lodging House / B&B': '#9E9E9E',
    'Exempt (not verified): Executive Suite': '#9E9E9E',
    'Exempt (verified)': '#607D8B',
    'Exempt (verified): Lodging House': '#607D8B',
    'Exempt (verified): Hospital Stays': '#607D8B',
    'Exempt (verified): Executive Suite': '#607D8B',
    'Exempt (verified): Bed & Breakfast Suite': '#607D8B',
    'Active': '#4CAF50',
    'Active (limit exceeded)': '#BF360C',
    'Expired/Void/Revoked': '#FF5722', 
    'Revoked': '#FF5722',
    'Void': '#FF5722',
    'Expired': '#FF5722',
    'Inactive': '#FF5722',
    'Not found (fabricated)': '#E64A19',
}

## UNIVERSAL ELEMENT ICONS [BY LICENSE STATUS]
status_icon_dict = {
    'Expired/Void/Revoked': 'fa-times-circle-o',
    'Expired': 'fa-times-circle-o',
    'Void': 'fa-times-circle-o',
    'Revoked': 'fa-times-circle-o',
    'Inactive': 'fa-times-circle-o',
    'No license claimed':  'fa-ban',
    'Not found (fabricated)': 'fa-exclamation-triangle',
    'Active': 'fa-check-square-o',
    'Active (limit exceeded)': 'fa-exclamation-triangle',
    'Exempt (verified)': 'fa-check-square-o',
    'Exempt (verified): Lodging House': 'fa-check-square-o',
    'Exempt (verified): Hospital Stays': 'fa-check-square-o',
    'Exempt (verified): Executive Suite': 'fa-check-square-o',
    'Exempt (verified): Bed & Breakfast Suite': 'fa-check-square-o',
    'Exempt (not verified)': 'fa-question-circle',
    'Exempt (not verified): Hospital Stays': 'fa-question-circle',
    'Exempt (not verified): Lodging House / B&B': 'fa-question-circle',
    'Exempt (not verified): Executive Suite': 'fa-question-circle',
}

# apply license colors to df
def color_license_status(df):
    global status_color_dict 
    if 'status_dict' in df.columns: # for centroids
        df['color'] = df.status_dict.apply(lambda x: status_color_dict[list(x.keys())[0]])
        return df
    else: # for listings
        df['color'] = df.status.map(status_color_dict)
        return df

# put STR buildings list on RAM
def load_adco_data(acdo_file='data_raw/STRbldgsGPSplus.csv'):
    print('—'*30)
    print(f'> Loading ADCO STR building database... {acdo_file}')
    acdo_data = pd.read_csv(acdo_file)
    acdo_data['latitude'] = acdo_data['latlong'].apply(lambda x: x.split(',')[0]).astype(float)
    acdo_data['longitude'] = acdo_data['latlong'].apply(lambda x: x.split(',')[1]).astype(float)
    print(f'  > {len(acdo_data)} STR buildings found')
    print('—'*30)
    return acdo_data


# load prepped data 
# (convert string JSONs to dict objects)
# (also calculate radius for each centroid in meters, for folium mapping)

def load_data():
    print('—'*60)
    # load cleaned, host-decoded data
    centroids_file = 'clustered/centroids.csv'
    listings_file = 'clustered/listings.csv'
    sub_c_file = 'clustered/sub_centroids.csv'
    config.adco_data = load_adco_data()


    centroids = pd.read_csv(centroids_file)
    sub_centroids = pd.read_csv(centroids_file)
    sub_centroids = pd.read_csv(sub_c_file)


    for c in centroids.columns:
        if '_dict' in c:
            try:
                centroids[c] = centroids[c].apply(lambda x:
                    json.loads( x.replace('"', "-999").replace("'", '"').replace('-999', "'") ) )
                sub_centroids[c] = sub_centroids[c].apply(lambda x:
                    json.loads( x.replace('"', "-999").replace("'", '"').replace('-999', "'") ) )
            except:
                print(f'ERROR CONVERTING COLUMN FROM DICT STRING TO JSON: {c}')
                centroids[c] = dict()
                sub_centroids[c] = dict()
            
    print(f'> Centroid data loaded from {centroids_file}')
    print('  >', centroids.host_id.nunique(), 'hosts |',
          centroids.centroid_group.nunique(), 'centroids |',
          centroids.count_listings.sum(), 'listings')

    print(f'> Sub centroid data loaded from {sub_c_file}')
    print('  >', sub_centroids.host_id.nunique(), 'hosts |',
          sub_centroids.sub_centroid.nunique(), 'centroids |',
          sub_centroids.count_listings.sum(), 'listings')

    listings = pd.read_csv(listings_file)
    for c in listings.columns:
         if '_dict' in c:
            listings[c] = listings[c].apply(lambda x: json.loads( x.replace("'", '"') ) )
    listings.rename(columns={'host_name':'host'}, inplace=True)
            
    listings.reset_index(drop=True, inplace=True)
    print(f'> Listings data loaded from {listings_file}')
    print('  >', listings.host_id.nunique(), 'hosts |',
        listings.centroid_group.nunique(), 'centroids |',
        len(listings), 'listings')
    print('—'*30)
    
    print('> Calculating centroids map element radius (alternate)') 
    centroids['radius2'] = 1.1*(centroids['avg_listing_distance'
                         ]+centroids['std_listing_distance']
                         )*0.3048 # convert feet -> meters for folium
    centroids.radius2 = centroids.radius2.apply(lambda x: max(80, x)) # set minimum centroid circle size  
    sub_centroids['radius2'] = 1.1*(sub_centroids['avg_listing_distance'
                         ]+sub_centroids['std_listing_distance']
                         )*0.3048 # convert feet -> meters for folium
    sub_centroids.radius2 = sub_centroids.radius2.apply(lambda x: max(80, x)) # set minimum centroid circle size  
    
    print('> Calculating centroids map element opacity') 
    scaler = MinMaxScaler()
    centroids['opacity'] = (scaler.fit_transform(centroids[['count_listings']].apply(np.log))*.2)+.4
    
    print('> Applying license status color key') 

    listings = color_license_status(listings)
    centroids = color_license_status(centroids)
    sub_centroids = color_license_status(sub_centroids)

    def pretty_groups(df):
        for feat in ['centroid_group', 'sub_centroid']:
            if feat in df.columns:
                df[feat] = df[feat].str.replace('_', '-')
        return df
    
    listings = pretty_groups(listings)
    centroids = pretty_groups(centroids)
    sub_centroids = pretty_groups(sub_centroids)
    
    return centroids, listings, sub_centroids # (centroids, listings)




# ———————————————————————————————————————————————————————————————
# —————— EXPANDED ADDRESS INFORMATION FOR CENTROIDS ————————————

# get "best address", geo dispersal, and other stats for each centroid on the map

def assess_isd_matches(centroids, listings):
    print('> Assessing ISD matches for centroids...')
    license_by_building_html = False
    
    # this loop is the entire function. centroid by cenroid
    for c_group in centroids.centroid_group.dropna().values:
        c_data = centroids[centroids.centroid_group==c_group].copy()
        c_row = c_data.index
        l_data = listings[listings.centroid_group==c_group].copy()

        found_addresses = l_data.ISD_address.notna().unique() # ISD address found?

        if l_data.ISD_address.notna().sum()==0: # NONE FOUND:
            # initialize list of exemptions or missings
            license_by_building_html = """
            <div class='license-by-building-container'>
                <ul>
                    """
            vcs = l_data.license.value_counts()
            for license, l_count in zip(vcs.index, vcs.values):
                license_status = l_data[l_data.license==license].status.values[0]
                            # for sidebar on centroid focus page
                license_by_building_html += f"""
                    <li>
                        <p>{license} - <b>{license_status}</b> ({l_count} listings)</b></p>
                    </li>
                """    
            license_by_building_html += """
                </ul>"""
            
            centroids.loc[c_row, 'ISD_bldg_details'] = license_by_building_html
            continue
            # end this centroid
            
            

    # AT LEAST SOME MATCHES...
        centroids.loc[c_row, 'ISD_found'] = True

        # PARTIAL FIND?
        found_count = l_data.ISD_address.isna().sum()
        if found_count:   # SOME FOUND
            not_found = l_data[l_data.ISD_address.isna()]
            centroids.loc[c_row, 'ISD_not_found'] = len(not_found) # CAN ACT AS A BOOL for a partial find
        # SOME ACTIVE, SOME NOT?
        inactive = (l_data.ISD_status!='Active').sum()
        if inactive:   # SOME FOUND
            inactive = l_data[l_data.ISD_status!='Active']
            centroids.loc[c_row, 'ISD_inactive'] = len(inactive) # CAN ACT AS A BOOL for a partial find

            
        # If any matches, create list
        found = l_data[l_data.ISD_address.notna()].copy()
        
        found['bldg_num'] = found.ISD_address.apply(lambda x: x.split()[0])
        
        license_by_building_html = """
                <div class='license-by-building-container'>
                    <ul>
        """
        
        # if multiple buildings...
        if found.bldg_num.nunique()>1:
            
            for bldg in found['bldg_num'].unique():
                bldg_data = found[found.bldg_num==bldg]
                bldg_adrs = bldg_data.ISD_address.values[0]
                units_found = bldg_data.ISD_address.nunique()
                
                
                # for sidebar on centroid focus page
                license_by_building_html += f"""
                    <li>
                        <p><b>{units_found} units</b> at {bldg_adrs}</p>
                        <ul>
                """           
                for license in bldg_data.license.dropna().unique():
                    listing_count = (bldg_data.license==license).sum()
                    isd_status = bldg_data[bldg_data.license==license].ISD_status.values[0]
                    license_by_building_html += f"""
                        <li>
                            <p>{license} - <b>{isd_status}</b> ({listing_count} listings) </p>
                        </li>
                    """
                license_by_building_html += """
                         </ul>
                    </li>"""
                
                # maybe generate html here
                   
            
            centroids.loc[c_row, 'ISD_match_type'] = f'In multiple buildings ({found.bldg_num.nunique()})'
            centroids.loc[c_row, 'ISD_bldg_details'] = license_by_building_html
            continue # keep to avoid overwriting bldg details!!!!

            
                
# if multiple units, 1 building...
        elif found.ISD_address.nunique()>1:    

            units_found = found['ISD_address'].nunique()
            
            centroids.loc[c_row, 'ISD_match_type'] = f'For multiple units ({units_found})'

# if single address...       
        else: 

            units_found = 1
            centroids.loc[c_row, 'ISD_match_type'] = f'At one address'
            

# for sidebar on centroid focus page (used if single building)
        license_by_building_html += f"""
            <li>
                <p><b>{units_found} unit{'s' if units_found!=1 else''}</b> at {found.ISD_address.values[0]}</p>
                <ul>
        """           
        for license in found.license.dropna().unique():
            listing_count = (found.license==license).sum()
            isd_status = found[found.license==license].ISD_status.values[0]
            license_by_building_html += f"""
                <li>
                    <p>{license} - <b>{isd_status}</b> ({listing_count} listings) </p>
                </li>
            """
        license_by_building_html += """
                 </ul>
            </li>"""
        centroids.loc[c_row, 'ISD_bldg_details'] = license_by_building_html
                 
    return centroids




# COMPILE DATA FOR GENERAL NEIGHBORHOOD of a location (from gcode)
def get_city_regions(centroids):
    print('> Getting city regions for centroids...')
    for row in centroids.index:

        region = []
        
        # neigh/suburb
        n = centroids.loc[row, 'GCODE_neighbourhood']
        if type(n)==str:
            region+=[n]
        s = centroids.loc[row, 'GCODE_suburb']
        if type(s)==str:
            region+=[s]
            
        if len(region)>1:
            centroids.loc[row, 'city_region'] = '/'.join(region)
            continue
            
        if len(region)>0:
            centroids.loc[row, 'city_region'] = '/'.join(region)
            continue
        
        # backup (probably never used)
        centroids.loc[row, 'city_region'] = 'Boston'
        
    return centroids
    
    
# COMPILE DATA FOR SPECIFIC ADDRESS (from ISD, or revert gcode)
def get_best_addresses(centroids, sub=False):
    print('> Getting best addresses for centroids...')
    for row in centroids.index:
                
        if not sub and centroids.loc[row, 'ISD_found']==True:
            centroids.loc[row, 'best_address_type'] = 'ISD'
            centroids.loc[row, 'best_address'] = list(centroids.loc[row, 'ISD_address_dict'].keys()
                                                     )[0].split('MA')[0].split('Boston')[0].strip()
            continue

        centroids.loc[row, 'best_address_type'] = 'GCODE'

        adrs_str = ''

        # house number
        h = centroids.loc[row, 'GCODE_house_number']
        if type(h)==str:
            adrs_str += h+' '

        # road or city block
        r = centroids.loc[row, 'GCODE_road']
        if type(r)==str:
            adrs_str += r
        elif 'GCODE_city_block' in centroids.columns:
            c = centroids.loc[row, 'GCODE_city_block']
            if type(c)==str:
                adrs_str += c
        if not sub:
            # suburb or neighborhood
            s = centroids.loc[row, 'GCODE_suburb']
            if type(s)==str:
                adrs_str += f', {s}' 
            else:
                n = centroids.loc[row, 'GCODE_neighbourhood']
                if type(n)==str:
                    adrs_str += f', {n}'

        centroids.loc[row, 'best_address'] = adrs_str
        
    centroids.best_address = centroids.best_address.str.replace(';', '/')
    return centroids
    
    
    
# define "top centroids" — aka centroids with n or more listings 
def big_centroid_focus(centroids, listings, min_size=5):
    config.min_top_centroid = min_size
    print(' ', '—'*30)
    print('  > Identifying big centroids...')
    l_olen = len(listings)
    c_olen = len(centroids)
    for c_group in centroids.centroid_group:
        c_idx = centroids[centroids.centroid_group==c_group].index        
        l_idx = listings[listings.centroid_group==c_group].index
        if len(l_idx) > min_size:
            centroids.loc[c_idx, 'high_confidence'] = True
            listings.loc[l_idx, 'high_confidence'] = True

    print(f'    > Would drop {c_olen - len(centroids)} centroids ({l_olen - len(listings)} listings)')
    print(f'    > Would remain: {len(centroids)} centroids ({len(listings)} listings)')
    print('—'*15)
    return centroids, listings 

        

    



# ———————————————————————————————————————— 
# ——— UNIVERSAL VARS & HELPER FUNCTIONS ——


# Locate nearby buildings in ADCO data for given centroid:
def isolate_probable_bldgs(ctrd_row):
    
    bldgs = config.adco_data
    lat_diff = abs(bldgs['latitude'] - ctrd_row.mean_latitude)
    lng_diff = abs(bldgs['longitude'] - ctrd_row.mean_longitude)
    
    bldgs['geo_diff'] = (lat_diff + lng_diff) / 2
    
    search_rad = .0005
    max_search_rad = .005
    nearby = bldgs[bldgs.geo_diff<search_rad].copy()
    while len(nearby)<1 and search_rad<max_search_rad:
        search_rad += .0002
        nearby = bldgs[bldgs.geo_diff<search_rad].copy()
        
    for row in nearby.index:
        
        bldg_lat = nearby.loc[row, 'latitude']
        bldg_lng = nearby.loc[row, 'longitude']
        distance = geodesic((ctrd_row.mean_latitude,ctrd_row.mean_longitude),
                            (bldg_lat,bldg_lng)).feet
        nearby.loc[row, 'centroid_distance'] = round(distance)
    
    return nearby
    
# create status indicator html for given centroid (or host!)
def get_status_indicator_html(c_data, size, text=True, tooltip=False, page_info=False):        
    
    status_dict = c_data.status_dict
    global status_color_dict
    global status_icon_dict

    
    icon_type = "fa-warning"
    color = ''
    message = ''
    icon_type = ''
    
    change_c = c_data.change_n_listings
    if str(change_c)=='nan':
        change_c = 0
        ref_msg = f'New location for host since {config.last_scraped}'
        ref_icon_type = 'fa-plus-square'
    elif change_c==0:
        ref_msg = f'No change detected since {config.last_scraped}'
        ref_icon_type = 'fa-square-o'
    elif change_c>0:
        ref_msg = f'{abs(int(change_c))} listings added since {config.last_scraped}'
        ref_icon_type = 'fa-plus-circle'
    elif change_c<0:
        ref_msg = f'{abs(int(change_c))} listings removed since {config.last_scraped}'
        ref_icon_type = 'fa-minus-circle' 
           
    # get status indicator message contents
    if len(status_dict)>1: # look for "WORST OFFENSE" by this cluster
        for status_text, status_count in status_dict.items():
             #multiple statuses?
            if status_text!= 'Active' and '(verified)' not in status_text:
                color = '#EF5350'
                message_n = status_count
                icon_type = status_icon_dict[status_text]
                if 'claimed' in status_text:
                    color = status_color_dict[status_text]
                break
        if color == '': # "WORST POSSIBLE" not found, search for next "worst"
            for status_text, status_count in status_dict.items():
                if status_text!= 'Active':
                    color = status_color_dict[status_text]
                    message_n = status_count
                    icon_type = status_icon_dict[status_text]
                    break
    else:
        status_text = list(status_dict.keys())[0] # only key
        color = status_color_dict[status_text]
        icon_type = status_icon_dict[status_text]
        message_n = status_dict[status_text]
        
    message = f"{status_text}"
    if not tooltip and not page_info:
        message += f"<br>({message_n} listings)"
        

    out = f"<p style='font-size:{size}; margin: 0 0 1px; color:{color}'>" # open p element
    
    
    # for FOLIUM MAP tooltip item
    if tooltip: 
        if color=='#9E9E9E': color = '#607D8B' # darker yellow
        if change_c!=0:
            out += f"""<span style='color:#448AFF!important;font-size:.92em'>
                <i class="fa {ref_icon_type}"></i> {ref_msg}</span> <br> """

        out += f"""<span style='color:black!important;font-size:.95em'>
                <b>Listings license status:</b></span><br>
            <i class="fa {icon_type}"></i> {message} """ 
        if c_data.shared_license==True: # if multiple licenses - **conditional close/open new p element**
            out += f"""
                </p><p style='font-size:.95; margin: 3px 0; color:#EF5350'>
                    <i class="fa fa-clone" aria-hidden="true"></i>
                    License shared by multiple hosts"""
        if type(c_data.license_exceed_type)==str: # if exceeded license category limit
            out += f"""
                 </p>
                <p style='font-size:.95; margin: 3px 0; color:#EF5350'>
                     <i class="fa fa-th-list" aria-hidden="true"></i>
                     License {c_data.license_exceed_type} limit exceeded
                    """
     # for centroid page sidebar detail
    elif text:
        if change_c!=0:
            out += f"""
        <span style='color:#448AFF'><i class="fa {ref_icon_type}"></i> {ref_msg}</span><br>
        """
        
        out += """
        <i class="fa {icon_type}"></i> {message}
            """
        
    # FOR SEARCH ITEM TOOLTIPS 
    elif not text:  # detected change in listings since last scrape
          
        t_i = 1
        if change_c!=0:
            out += f"""
            <div class="info-tooltip">
                <i style='color:#448AFF !important' class="fa {ref_icon_type}" aria-hidden="true"></i>
              <span style='color:#448AFF' class="tooltiptext">
                  {ref_msg}
              </span>
            </div>
                """
            t_i += 1 # tooltip counter (for margin setting)
            
        out += f"""
        <div class="info-tooltip" style='color:{color}'>
            <i class="fa {icon_type} "></i>
            <span style='color:{color}' class="tooltiptext tooltiptext-{t_i}">
                {message}</span>
        </div>
            """
        t_i+=1
        # add more popups if additional warnings
        if c_data.shared_license==True: # if multiple licenses
            out += f"""
                    <div class="info-tooltip">
                    <i style='color:#EF5350 !important' class="fa fa-clone" aria-hidden="true"></i>
                          <span style='color:#EF5350' class="tooltiptext tooltiptext-{t_i}">
                              License shared by multiple hosts
                          </span>
                    </div>
                    """
    out += "</p>"
    
    return out


# CREATE THE GLOBAL SEARCH BAR (on home and high-conf centroids pages)
def global_search(centroids, listings):
    # COMPILE SEARCH TERMS (UNIVERSAL) (with **all** interesting centroids)

    search_html = '' # to compile into
    
    # COMPILE HOSTS
    print('—'*30)
    print('  > Creating global search bar...')
    for host in tqdm(listings.host_id.value_counts().index):
        
        h_data = listings[listings.host_id==host]
        
        host_name = h_data.host.values[0]
        host_id = h_data.host_id.values[0]
        licenses = list(h_data.license.unique())
        aliases = list(h_data.host_alias.dropna().unique())
        alias_ids = list(h_data.alias_id.unique())
        
        h_clusters = h_data.centroid_group.unique()
        hc_data = centroids[centroids.host_id==host]       

        search_html += f"""
        <li onclick="location.href='/host/{host}.html'"
            class="search-items"
            data-type="host"
            search-data = "
                     Host {host_name} | 
                     Aliases: {aliases} |
                     Clusters: {h_clusters}
                     ID(s): {host_id} Alias Id(s): {alias_ids} |
                     License(s): '{licenses}'">
            <p style='font-size:1em;color:white'>Host: <b style='color:#C5B358'>{host_name}</b></p>
            {f"<p style='font-size:.9em;color:white'>Aliases: <b>{', '.join(aliases)}</b></p>" if len(aliases) else ''}
            <p style='font-size:.9em; color:white'>Clusters: <b>{len(h_clusters)}</b> | Listings: <b>{len(h_data)}</b></p>
        </li>
        """
        

    # ADD EACH CLUSTER
    for row in centroids.index:
        c_data = centroids.loc[row]
        status_indicator_html = get_status_indicator_html(c_data, size='.85em', text=False)
        aliases = list(c_data['alias_dict'].keys())
        l_data = listings[listings.centroid_group==c_data.centroid_group]
        licenses = list(l_data.license.unique())
        ids = list(l_data['id'].unique())
    
    
        # include additional search terms in element DOM under CSS selector "search-data"
        search_html += f"""
        <li data-type="cluster" onclick="location.href='/cluster/{c_data.centroid_group}.html'"
                        class="search-items"
            data-type="host"
            search-data = "
                     Cluster {c_data.centroid_group} |
                     Host: {c_data.host} | 
                     Aliases: {aliases} |
                     Cluster: {c_data.centroid_group}
                     Host ID: '{c_data.host_id}' |
                     License(s): '{licenses}' |
                     Listing(s): '{ids}' |
                     Address(es): '{c_data.best_address}' | 
                     Neighborhood(s): '{c_data.city_region} {list(l_data.neighbourhood_cleansed.unique())}' |">
                     
            <div class='isd-indicator-search'>{status_indicator_html}</div>
            <div stlye='padding:0 35px 0 0'>
                <p style='font-size:1em;color:white'>Cluster <b style='color:#448AFF'>{c_data.centroid_group}</b></p>
                <p style='font-size:.9em; color: white; margin-bottom:4px'>
                    Host: <b>{c_data.host}</b> | Listings: <b>{c_data.count_listings}</b></p>
                <p style='font-size:.9em; color: white;  margin-bottom:4px'>
                    {c_data.best_address.split('Boston')[0]}</p>
        """
        search_html += "</div></li>"
        
        
    config.search_html = f"""
                
                <div id="search-container">
                    <input autocomplete="false" id="searchbar" onkeyup="search_item()" tabindex="1"
                    type="text" name="search" placeholder="Search...">
                    
                    <div id='search-message-container'> 
                        <p id="search-instruct" style="display: block; font-size:.88em !important;
                            color:#448AFF;">
                                click for more details:</p>
                        <p id="search-alert" style="display: block; font-size:.9em !important;
                                color:#F44336;">
                                    <b>No results</b></p>
                        <p id="search-init" style="display: block; font-size:.9em !important;
                                color:white;">
                                    <em>Search by <b>host, listing ID, cluster or neighborhood</b></em></p>
                    </div>
                    <ul id="search-list">
                        {search_html}
                    </ul>
               </div>

    """
    return



    
# ———————————————————————————————————————— 
# ————————— TOOLTIPS + POPUPS ———————————— 

# (tooltip = hover) (popup = click)


def centroid_toolip(c_data, listings, more_button=True, sub=False):
    multi_license = len(c_data.license_dict.items())>1
    if multi_license: license_list = 'Licenses (multiple):'
    else: license_list = 'License:'
    
        
    t = f"""
        <p style='font-size:1.25em; margin:0; color:#448AFF'>
        {'Cluster <b>'+c_data.centroid_group if not sub else '(Sub)cluster <b>'+c_data.sub_centroid}
        </b></p>
        {get_status_indicator_html(c_data, size='1.0em', tooltip=True)}
        Host: <b>{c_data.host} </b> | Listings: <b>{c_data.count_listings}</b>
        <br>Minimum  stay:  <b style='color:{'#FF5722' if c_data.minimum_nights<28 else 'black'}'>
            {int(c_data.minimum_nights)} night{'s' if c_data.minimum_nights!=1 else ''} </b>  
        
        """
    if sub and len(str(c_data.best_address))>3:
        t += f"<br>At or near:<br><b style='color:#448AFF'>{c_data.best_address}</b></span>"
    if more_button:
        t += "<br><span style='text-align:center; color:#448AFF'><b>Click for details</b></span>"
            
    return folium.Tooltip(t)

def centroid_popup(c_data):
    c_group = c_data.centroid_group
    h_id = c_data.host_id
    host = c_data.host

    if c_group == 'None':
        p = folium.Popup(f"""<div style="white-space: nowrap; text-align:center; color:white">
                                <b style='font-size:14px;'>Cluster not assigned</b>
                                <br>
                                <b style='font-size:12px'>(Host has fewer than 3 listings)</b>                            
                        </div>
                        """)
    elif str(c_group) == 'nan':
        p = folium.Popup(f"""<div style="white-space: nowrap; text-align:center; color:white">
                                <b style='font-size:14px'>Cluster not assigned</b>
                                <br>
                                <b style='font-size:12px'>(Not enough host listings nearby)</b>                            
                        </div>
                        """)
    else:
        p = folium.Popup(f"""<div style="white-space: nowrap; text-align:center">
                            <a class='centroid-link-btn'
                               href='/cluster/{c_group}.html'>
                                <b>Cluster {c_group}</b> |
                                Details <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            
                        </div>
                        """)
    return p


def sub_centroid_popup(c_data):
    
    sub_c_idx = c_data.sub_centroid
    p = folium.Popup(f"""<div style="white-space: nowrap; text-align:center">
                        <a class='centroid-link-btn'
                           href='/sub_cluster/{sub_c_idx}.html'>
                            <b>(Sub)cluster {sub_c_idx}</b> |
                            Details <i class="fa fa-arrow-right" aria-hidden="true"></i>

                    </div>
                    """)
    return p

def listing_toolip(l_data, more_button=False):

    t = f"""
        <p style='font-size:1.25em; margin:0; color:#BF0000'>Listing <b>{l_data.id}</b></p>
        Cluster: <span style='color:#448AFF'><b>{l_data.centroid_group}</b></span> | 
        
        Host: <span style='color:#665c2c'><b>{l_data.host} </b></span>
        """

    
    if str(l_data.host_alias)!='nan' and str(l_data.host_alias)!=l_data.host: 
        t += f"""
        <br>Alias account: <span style='color:#636363'><b>{l_data.host_alias} </b></span>
        """
    t += f"""
        <br>License:
                 <i style='font-size:.9em; color:{l_data.color}';
                       class='fa fa-file-text'></i>
                <b> {l_data.license}</b>
        <br>License: <span style='color:{l_data.color}'> <b>{l_data.status}</b> </span>
        """
        
    if more_button:
        if str(l_data.centroid_group)!='nan':
            t += f"""
                    <br><b style='color:#448AFF'>Click or see cluster for more details</b></span>"""
        else:
            t += f"""
                    <br><b>Outlier not assigned to a cluster</b></span>"""
    return folium.Tooltip(t)



def adco_tooltip(row):
    
    
    html = "<b style='font-size:.88em'>Known STR Building</b><br>"
    
    if str(row['Bldg name'])!='nan':
        html += f"{row['Bldg name']}<br>"
        
    html += "<b style='font-size:1.2em; color:#5E35B1'>"+str(row.GPSaddress)+'</b><br>'
    
    details = []
    row = row.fillna('(unknown)')
    if row['Bldg units']: details.append(f"Units: {row['Bldg units']}")
    if row['Bldg flrs']: details.append(f"Floors: {row['Bldg flrs']}")
    if row['Source']: details.append(f"Source: {row['Source']}")
    html+= ' | '.join(details)+'<br>'
        
    bldg_hosts = []
    for col, val in zip(row.index, row.values):
        if 'Bldg' not in col and str(val)==str(1.0): # de-pivot host list table
            bldg_hosts.append(col)
            
    if len(bldg_hosts)<1: bldg_hosts.append('unknown')
    html += "<b>Known operators: </b><br>"+'<br>'.join([' &#8226; ' + b for b in bldg_hosts])
    
    html += "<div style='height:4px'></div>"
    html += f"<span style='color:#448AFF' >Distance from cluster center: <b>{int(row.centroid_distance)} ft.</b></span>"
    
    return folium.Tooltip(html)






# ———————————————————————————————————————— 
# ———————————— MAP ANNOTATIONS ———————————

def add_updated(m, listings):
    updated = f"""
         <div style="
         position: fixed; z-index:699; 
         bottom: 16px; right: 0; width: auto; height: auto; 
         background-color:white;
         border-bottom:1px solid #DEDEDE;
         opacity: .7;
         padding: 3px 8px;
         font-size:12px">
            <span style='margin: 0px;'>
                <a style='color: black !important' target='blank' href='insideairbnb.com'>InsideAirbnb</a>:
                    {config.latest_scraped} | STR registry: {config.ISD_updated}
            </span>
          </div> """
    m.get_root().html.add_child(folium.Element( updated ))
    return m






# ———————————————————————————————————————— 
# ——————————— WEB PAGE SIDEBAR ———————————

# all of this is for the license status legend (in sidebar):
def get_status_vcs(data, spec=False): # accepts list of listings for target group
    
    global status_color_dict
    global status_icon_dict
    if spec:
        vcs = data.simple_status.value_counts().sort_index() # (simple status)
    else:
        vcs = data.full_status.value_counts().sort_index() # (simple status)
    status_order_dict = {v: i for i, v in enumerate(list(status_color_dict.keys())) }
    
    vcs = data.full_status.value_counts().sort_index()
    vcs_norm = data.full_status.value_counts(True).sort_index()

    out = vcs.to_frame().reset_index()
    out.columns = ['status', 'count']
    out['pct'] = vcs_norm.values
    out['color'] = out['status'].map(status_color_dict)
    out['icon-type'] = out.status.map(status_icon_dict)
    out = out.sort_values('status', key= lambda x: [status_order_dict[i] for i in x]  )
    return out

# make overall status summary HTML for host or centroid group
def make_status_legend(listings, spec=False, page_type='none'): 
    
    status_vcs = get_status_vcs(listings, spec)
    
    
    status_key_html = f"""<table class='listings-status-legend'>
                            
                            <tr style='border-bottom:1px solid #9c9c9c;' >
                                <td>  
                                   <p class='text-lite'
                                      style='margin-bottom:8px'>
                                       <b>listings</b><p>
                                </td>
                                
                                <td>  
                                </td>
                                                                
                                <td>
                                    <p class='text-lite'
                                       style='margin-bottom:8px'>
                                        <b>license</b>
                                    </p>
                                </td>
                                """
    if page_type=='home':
        status_key_html+= """
                                <td>
                                    <p class='text-lite'
                                          style='margin-bottom:8px'>
                                           <b>pct</b>
                                    </p>
                                </td>  
                                """
    status_key_html += "</tr>"
    
    
    for row in status_vcs.index:
        status_info = status_vcs.loc[row]
        row_html = f"""                           
                            <tr>
                                
                                <td style='text-align:right'>  
                                    <p>
                                    {status_info['count']} x</p> 
                                </td>
                                
                                <td>                                      
                                <i class='fa {status_info['icon-type']}'
                                   style='color:{status_info['color']};
                                   margin: 0 3px; font-size:16px'></i>
                                </td>
                                
                                <td><p>
                                    {status_info.status}
                                </p></td>"""
                                
        if page_type=='home':
            pct_val = status_info.pct
            row_html += f"""
                                <td>
                                    <p>
                                        {round((pct_val*100), 1)}%
                                    </p>
                                </td>  """
            
        row_html += "</tr>"   
        

        status_key_html += row_html
        
    return status_key_html + '</table>'





# ———————————————————————————————————————————————————————————————————————————————————————————————————— 
# ——————————————————————————————————— MAP HTML  FUNCTIONS  ———————————————————————————————————————————

      

def map_top_centroids(centroids, listings):
    
    centroids = centroids[centroids.high_confidence==True].copy()
    listings = listings[listings.high_confidence==True].copy()

    start_coords = [42.343, -71.085]
    zoom_lvl = 13     
    m = folium.Map(location=start_coords, tiles=None, zoom_start=zoom_lvl)
    folium.TileLayer('cartodbpositron', control=False).add_to(m) # hide "all" button
    
    # ADCO BUILDINGS
    bldg_fg = folium.FeatureGroup( 
                        name="""<span style='font-size:15px'>
                        Known STR buildings near large clusters
                        <i class='fa fa-building'
                           style='color:#5E35B1; margin: 0 3px;'></i>
                        </span>""",
                show=False,
                control=True) # for user layer control
    m.add_child(bldg_fg)
    

    # CENTROID AREA CIRCLES
    for row in centroids.sort_values('radius2', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 2,
                            fill_color = c_data.color,
                            radius =  float(c_data.radius2),
                            opacity = float(c_data.opacity+.3),
                            fill_opacity = c_data.opacity,
                            tooltip = centroid_toolip(c_data, listings, more_button=True),
                            popup = centroid_popup(c_data),
            ).add_to(m)

        # nearby ADCO buildings for this centroid
        bldgs = isolate_probable_bldgs(c_data)
        for b_row in bldgs.index:
            b_data = bldgs.loc[b_row]
            folium.Marker([b_data.latitude, 
                        b_data.longitude],
                        icon=folium.Icon(color='lightgray',
                                         icon_color='#5E35B1',
                                         icon='building',
                                         prefix='fa'),
                        tooltip = adco_tooltip(b_data),
            ).add_to(bldg_fg) 
        
        
# CENTROID CENTER MARKERS
    for row in centroids.sort_values('radius', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 4,
                            fill_color = '#448AFF',
                            radius =  6,
                            opacity = .7,
                            fill_opacity = 1,
                            tooltip = centroid_toolip(c_data, listings, True),
                            popup = centroid_popup(c_data),
            ).add_to(m)
            

    # LISTING MARKERS
    listing_fg = folium.FeatureGroup( 
            name="""<span style='font-size:15px'>
                        Include individual listings on map
                        <i class='fa fa-dot-circle-o'
                           style='color:red; margin: 0 3px;'></i>
                    </span>""",
            show=(False),
            control=True) # for user layer control
    m.add_child(listing_fg)
    for row in listings.index:
        l_data = listings.loc[row]
        folium.vector_layers.Circle(
                            [l_data.latitude, 
                             l_data.longitude],
                            color = 'red',
                            fill_opacity = .8,
                            weight=2.5,
                            opacity = .9,
                            fill_color = l_data.color,
                            radius = 10,
                            tooltip = listing_toolip(l_data, True),
                            popup = centroid_popup(l_data),
            ).add_to(listing_fg)

                    
    folium.LayerControl(collapsed=False).add_to(m)
    
    m = add_updated(m, listings)
    return m





# FOR HOME PAGE:
def map_listings(listings): # based on input, detects the target group type

    # base map
    start_coords = [42.315, -71.075]
    zoom_lvl = 12
    m = folium.Map(location=start_coords, tiles=None, zoom_start=zoom_lvl) 
    folium.TileLayer('cartodbpositron', control=False).add_to(m) # hide "all" button
    
    status_order_dict = {v: i for i, v in enumerate(list(status_color_dict.keys())) }
    sorted_vcs = listings.status.drop_duplicates(
                        ).sort_values(key = lambda x: [status_order_dict[i] for i in x] )
    for status in sorted_vcs.values:
        status_data = listings[listings.status==status].copy()
        globals()[f'{status}_fg'] = folium.FeatureGroup(
            name=status,
            control=True,
            show = True )
        globals()[f'{status}_fg'].add_to(m)
        for row in status_data.index:
            l_data = status_data.loc[row]
            folium.vector_layers.Circle(
                                [l_data.latitude, 
                                 l_data.longitude],
                                color = l_data.color,
                                fill_opacity = .5,
                                weight = 3,
                                opacity = 1,
                                fill_color = l_data.color,
                                radius = 18,
                                tooltip = listing_toolip(l_data, True),
                                popup = centroid_popup(l_data),
                ).add_to(globals()[f'{status}_fg'])

    folium.LayerControl(collapsed=False).add_to(m)
    m = add_updated(m, listings)
    return m


# CREATE THE MAP WITH THE GIVEN DATA. 
def map_focus_data(feature, group, centroids, listings, sub_centroids):
    
    centroids = centroids[centroids[feature]==group].copy()
    listings = listings[listings[feature]==group].copy()
    sub_centroids = sub_centroids[sub_centroids[feature]==group].copy()
    start_coords = [
        (listings.latitude.max()-listings.latitude.min()
                )/2 + listings.latitude.min(),
        (listings.longitude.max()-listings.longitude.min()
                )/2 + listings.longitude.min(),]

    
    # set custom zoom level based on listing dispersalL

    lng_range =  listings.longitude.max() - listings.longitude.min()
    lat_range = listings.latitude.max() - listings.latitude.min()
    if 'centroid' in feature:
        if max(lat_range, lng_range) > .004:
            zoom_lvl = 16
        if max(lat_range, lng_range) > .0012:
            zoom_lvl = 17
        else:
            zoom_lvl = 18
    elif 'host' in feature:
        zoom_lvl = 16
        zoom_add = 0
        
        if lat_range > .002:
            zoom_add = max(1, zoom_add)
        if lat_range > .015:
            zoom_add = max(2, zoom_add)
        if lat_range > .03:
            zoom_add = max(3, zoom_add)
        if lat_range > .07:
            zoom_add = max(4, zoom_add)

        if lng_range > .002:
            zoom_add = max(1, zoom_add)
        if lng_range > .03:
            zoom_add = max(2, zoom_add)
        if lng_range > .068:
            zoom_add = max(3, zoom_add)
        if lng_range > .1:
            zoom_add = max(4, zoom_add)
            
        zoom_lvl -= min(4, zoom_add)
        
    m = folium.Map(location=start_coords, tiles=None, zoom_start=zoom_lvl)
    folium.TileLayer('cartodbpositron', control=False).add_to(m) # hide "all" button
    
    
    # ADCO BUILDINGS
    bldg_fg = folium.FeatureGroup( 
                        name="""<span style='font-size:15px'>
                        Known STR buildings near large clusters
                        <i class='fa fa-building'
                           style='color:#5E35B1; margin: 0 3px;'></i>
                        </span>""",
                show=True,
                control=True) # for user layer control
    m.add_child(bldg_fg)
    

    # CENTROID AREA CIRCLES
    for row in centroids.sort_values('radius2', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 2,
                            fill_color = c_data.color,
                            radius =  float(c_data.radius2),
                            opacity = float(c_data.opacity+.3),
                            fill_opacity = c_data.opacity,
                            tooltip = centroid_toolip(c_data, listings,
                                                      more_button=True if 'host' in feature else False),
                            popup = centroid_popup(c_data) if 'host' in feature else None,
            ).add_to(m)
        
    # SUB CENTROID AREA CIRCLES
    for row in sub_centroids.sort_values('radius2', ascending=False).index:
        c_data = sub_centroids.loc[row]
        if config.sub_mode:
            folium.vector_layers.Circle(
                [c_data.mean_latitude, 
                 c_data.mean_longitude],
                color = c_data.color,
                fill = True,
                weight = 3,
                fill_color = '#448AFF',
                radius =  float(c_data.radius),
                opacity = .9,
                fill_opacity = .2,
                tooltip = centroid_toolip(c_data, listings,
                                          more_button=True if 'host' in feature else False,
                                          sub=True),
                popup = sub_centroid_popup(c_data) if 'centroid' in feature else centroid_popup(c_data),
                ).add_to(m)

        # nearby ADCO buildings for this centroid
        bldgs = isolate_probable_bldgs(c_data)
        for b_row in bldgs.index:
            b_data = bldgs.loc[b_row]
            folium.Marker([b_data.latitude, 
                        b_data.longitude],
                        icon=folium.Icon(color='lightgray',
                                         icon_color='#5E35B1',
                                         icon='building',
                                         prefix='fa'),
                        tooltip = adco_tooltip(b_data),
            ).add_to(bldg_fg) 
        
        
# CENTROID CENTER MARKERS
    for row in centroids.sort_values('radius', ascending=False).index:
        c_data = centroids.loc[row]
        folium.vector_layers.Circle(
                            [c_data.mean_latitude, 
                             c_data.mean_longitude],
                            color = '#448AFF',
                            fill = True,
                            weight = 4,
                            fill_color = '#448AFF',
                            radius =  6,
                            opacity = .7,
                            fill_opacity = 1,
                            tooltip = centroid_toolip(c_data, listings,
                                                      more_button=True if 'host' in feature else False),
                            popup = centroid_popup(c_data) if 'host' in feature else None,
            ).add_to(m)
            

    # LISTING MARKERS
    listing_fg = folium.FeatureGroup( 
            name="""<span style='font-size:15px'>
                        Include individual listings on map
                        <i class='fa fa-dot-circle-o'
                           style='color:red; margin: 0 3px;'></i>
                    </span>""",
            show=True,
            control=True) # for user layer control
    m.add_child(listing_fg)
    for row in listings.index:
        l_data = listings.loc[row]
        folium.vector_layers.Circle(
                            [l_data.latitude, 
                             l_data.longitude],
                            color = 'red',
                            fill_opacity = .8,
                            weight=2.5,
                            opacity = .9,
                            fill_color = l_data.color,
                            radius = 10,
                            tooltip = listing_toolip(l_data, True),
                            popup = centroid_popup(l_data),
            ).add_to(listing_fg)
        

    folium.LayerControl(collapsed=False).add_to(m)
    m = add_updated(m, listings)
    return m


# FOR MAPPING ONE SUB-CENTROID ——— CURRENTLY DISABLED IN PUBLIC RELEASE
def map_one_sub(group, listings, sub_centroids):
    feature = 'sub_centroid'
    
    listings = listings[listings[feature]==group].copy()
    sub_centroids = sub_centroids[sub_centroids[feature]==group].copy()

    start_coords = [
        (listings.latitude.max()-listings.latitude.min()
                )/2 + listings.latitude.min(),
        (listings.longitude.max()-listings.longitude.min()
                )/2 + listings.longitude.min(),]

    lng_range =  listings.longitude.max() - listings.longitude.min()
    lat_range = listings.latitude.max() - listings.latitude.min()
    
    if max(lat_range, lng_range) > .004:
        zoom_lvl = 16
    if max(lat_range, lng_range) > .0012:
        zoom_lvl = 17
    else:
        zoom_lvl = 18
        
    m = folium.Map(location=start_coords, tiles=None, zoom_start=zoom_lvl)
    folium.TileLayer('cartodbpositron', control=False).add_to(m) # hide "all" button
    
    
    # ADCO BUILDINGS
    bldg_fg = folium.FeatureGroup( 
                        name="""<span style='font-size:15px'>
                        Known STR buildings near large clusters
                        <i class='fa fa-building'
                           style='color:#5E35B1; margin: 0 3px;'></i>
                        </span>""",
                show=True,
                control=True) # for user layer control
    m.add_child(bldg_fg)
    
        
    # SUB CENTROID AREA CIRCLES
    
    for row in sub_centroids.sort_values('radius2', ascending=False).index:
        c_data = sub_centroids.loc[row]
        if config.sub_mode:
            folium.vector_layers.Circle(
                [c_data.mean_latitude, 
                 c_data.mean_longitude],
                color = c_data.color,
                fill = True,
                weight = 3,
                fill_color = '#448AFF',
                radius =  float(c_data.radius),
                opacity = .9,
                fill_opacity = .2,
                tooltip = centroid_toolip(c_data, listings,
                                          more_button=True if 'host' in feature else False,
                                          sub=True),
                popup = sub_centroid_popup(c_data) if 'host' in feature else None,
            ).add_to(m)

        # nearby ADCO buildings for this centroid
        bldgs = isolate_probable_bldgs(c_data)
        for b_row in bldgs.index:
            b_data = bldgs.loc[b_row]
            folium.Marker([b_data.latitude, 
                        b_data.longitude],
                        icon=folium.Icon(color='lightgray',
                                         icon_color='#5E35B1',
                                         icon='building',
                                         prefix='fa'),
                        tooltip = adco_tooltip(b_data),
            ).add_to(bldg_fg) 
        

    # LISTING MARKERS
    listing_fg = folium.FeatureGroup( 
            name="""<span style='font-size:15px'>
                        Include individual listings on map
                        <i class='fa fa-dot-circle-o'
                           style='color:red; margin: 0 3px;'></i>
                    </span>""",
            show=True,
            control=True) # for user layer control
    m.add_child(listing_fg)
    for row in listings.index:
        l_data = listings.loc[row]
        folium.vector_layers.Circle(
                            [l_data.latitude, 
                             l_data.longitude],
                            color = 'red',
                            fill_opacity = .8,
                            weight=2.5,
                            opacity = .9,
                            fill_color = l_data.color,
                            radius = 10,
                            tooltip = listing_toolip(l_data, True),
                            popup = centroid_popup(l_data),
            ).add_to(listing_fg)
        
        
    folium.LayerControl(collapsed=False).add_to(m)
    
    m = add_updated(m, listings)
    return m



# FOR MAPPING ONE CENTROID
def map_single_centroid(centroids, listings, sub_centroids):
    
    # base ma        
    if auto_zoom=='dynamic': # dynamic map positioning
        zoom_lvl = 16
        zoom_add = 0        
        start_coords = [
            (centroids.mean_latitude.max()-centroids.mean_latitude.min()
                    )/2 + centroids.mean_latitude.min(),
            (centroids.mean_longitude.max()-centroids.mean_longitude.min()
                    )/2 + centroids.mean_longitude.min(),]
        
        lng_range =  centroids.mean_longitude.max() - centroids.mean_longitude.min()
        lat_range = centroids.mean_latitude.max() - centroids.mean_latitude.min()
        if lat_range > .002:
            zoom_add = max(1, zoom_add)
        if lat_range > .015:
            zoom_add = max(2, zoom_add)
        if lat_range > .03:
            zoom_add = max(3, zoom_add)
        if lat_range > .07:
            zoom_add = max(4, zoom_add)
        
        if lng_range > .002:
            zoom_add = max(1, zoom_add)
        if lng_range > .03:
            zoom_add = max(2, zoom_add)
        if lng_range > .068:
            zoom_add = max(3, zoom_add)
        if lng_range > .1:
            zoom_add = max(4, zoom_add)
        
        zoom_lvl -= min(4, zoom_add)


        
        
### PAGE HTML GENERATOR FUNCTIONS


# ONE FUNCTION TO GENERATE UNIVERSAL STYLE & SCRIPT:

def gen_assets(page_type='none'): # removes hover anim for centroids if on centroid page
    

    # HOME PAGE CSS
    sidebar_width = '400px' # var
    style = """
    
    
/* Tooltip container */
.tooltip {
  opacity: 1;
  position: relative;
  display: inline-block;
  font-family: "Helvetica", Arial, sans-serif;
}

.tooltiptext-2 {
    margin-top: 25px !important;
}

.tooltiptext-3 {
    margin-top: 46px !important;
}

/* Tooltip text */
.tooltiptext {
  visibility: hidden;
  width: auto;
  color: #fff;
  text-align: right;
  white-space: nowrap;
  padding: 3px 8px;
  border-radius: 5px;
  background-color : #424242;
  /* Position the tooltip text */
  margin: 3px 0 0 0 ;
  position: absolute;
  z-index: 1;
  top: -5px;
  right: 105%;

  /* Fade in tooltip */
  opacity: 0;
  transition: opacity 0.2s;
}



/* Show the tooltip text when you mouse over the tooltip container */
.info-tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
    
    html {
      scroll-behavior: smooth;
        height: -webkit-fill-available;

    }
    
    body {
        font-family: "Helvetica", Arial, sans-serif;
        padding:0;
        margin:0;
        height:100vh;
        max-height: -webkit-fill-available;
        overflow: auto;
        color: white;
    }
    
    .leaflet-popup-content-wrapper, .leaflet-popup-tip {
        background-color:#448AFF !important;
        }
        
    .leaflet-popup-close-button {
        display: none !important
    }
    """
    
    if page_type!='cluster':
        style += """
    
        .leaflet-pane .leaflet-interactive {
            transition: fill-opacity .1s ease-in-out;
            transition: stroke .1s ease-in-out;

        }

        .leaflet-pane .leaflet-interactive:hover {
            fill-opacity:.8 !important;
            stroke:white !important;
        }
        """

    style += """
    .centroid-link-btn {
        color:white !important;
        font-size:18px;
        padding:15px 0;
        line-height:1.2em
        }
        


    body a {
        color: #E3F2FD;
        text-decoration:none;
        transition:.1s;
    }
    body a:hover {
        color: #448AFF;
        text-decoration: underline
    }
    
    body a:active {
        color: #D4D4D4;
    }

    body h1 {
         color: white;
         font-size:24px;
         margin-top:0;
    }

    body h2 {
        color:#448AFF;
        font-size:22px;
        line-height:1.4em;
        font-weight:400px;
        margin: 8px 0 8px;
        font-weight:bold;
    }

    body h3 {
        color:white;
        font-size:18px;
        margin: 0 0 8px;
        font-weight:400px;
        font-weight:500px;
    }

    body h4 {
        font-size:16px;
        color:white;
        margin: 0 0 10px;
        font-weight:500px;
    }

    body p {
        font-size:16px;
        line-height:1.4em;
        color:white;
        margin: 0 0 8px;
    }

    body .text-lite {
        color:#E0E0E0;
        font-size: .92 em;
    }

    hr {
        border: .6px solid white;
        margin: 15px 0
        style='margin:20px'
    }
    
    .hr-lite {
        border-color: #9c9c9c;
    }


    #page-container {
        padding: 0;
        display:block;
        height: 100%;
        width: 100vw;
        box-sizing: border-box;
    }

    #sidebar-container {
        position:fixed;
        display: block;
        height: 100vh;
        height: -webkit-fill-available;
        width:"""+sidebar_width+""";
        overflow:auto;
        transition:.2s;
        z-index: 9999;
    }

    #nav-banner {
        position:fixed;
        top:0;
        left:0;
        z-index:9;
        margin:0;
        width:"""+sidebar_width+""";
        background-color:#191919;
        overflow:auto;
        z-index: 10000;
    }

    #sidebar {
        width:"""+sidebar_width+""";
        top: 25px;
        position:fixed;
        height:calc(100vh - 25px);
          max-height: -webkit-fill-available;

        display:block;
        background-color: #2b2b2b;
        #border: 5px solid #4a4a4a;
        box-sizing: border-box;
        overflow-y: auto;
        padding:30px;
        transition:.15s;
    }
    
    #open-menu-icon {
        visibility:hidden;
        display:block;
        position:fixed;
        left:0;
        top:48vh;
        color:white;
        padding:10px;
        font-size: 1.5em;
    }
    
    
    #mobile-switcher {
        visibility:hidden;
        position: absolute;
        right: 0;
        top: 0;
        display:inline-block;
        padding: 5px;
        background-color: #191919;
        margin:0;
        pointer:cursor;
    }
    

    # mobile-switch-label {
        margin: 2px 0 0;
    }
    
    #mobile-switch-link {
        display:block;
        position:fixed;
        top:40px;
        right:0;
        width:30px;
        height: calc(100vh - 40px);
  max-height: -webkit-fill-available;
        z-index:99999;
        visibility:hidden;
        pointer:cursor;
    }
    
    @media screen and (max-width: 850px){
    
            .modal {
            padding:0 !important;

        }
    
        #mobile-switcher {
            visibility: visible;
        }
        #mobile-switch-link {
            visibility: visible;
        }
        #nav-banner {
            width: 100vw;
            padding:5px;
        }
        #sidebar-container {
            width:calc(100vw - 30px);
        }
        #sidebar-container {
            box-shadow: 0 0 20px rgba(0,0,0,0.8);
        }
        #sidebar {
            width:calc(100vw - 30px);
        }

        #map-container, iframe {
            top:40px !important;
            height:calc(100vh - 40px)  !important;
              max-height: -webkit-fill-available;

            z-index: 9995;
            width:calc(100vw - 30px) !important;
        }       
    }
    
    
    @media screen and (max-width: 720px){

        .leaflet-control-layers {
            display:none;
        }
    
    }
    
    #mobile-menu-circle {
      height: 65px;
      width: 65px;
      background-color: #2b2b2b;
      border-radius: 50%;
      display: inline-block;
      position: fixed;
      text-align: right;
      left: -100px;
      padding: 15px 3px;
      transition:.2s
    }
    
    .mobile-menu-circle i {
        display: table-cell;
        width: 65px;
        height: 65px;
        padding: 5px;
        border: none;
        vertical-align: middle;
        
    }
    
    
    #nav-banner p {
        margin:4px 8px;
        width:max-content;
        padding
    }

    .listings-status-legend {
        margin-bottom:12px;
        padding: 0 15px
    }
    .listings-status-legend p {
        margin:0;
    }
    .listings-status-legend td {
        padding: 10px 5px 0;
    }
    
    .listings-status-legend th {
        font-size:1.2em;
        padding: 10px 5px 0;
        color: white !important
    }

    #status-icon {
        height:15px;
        width:15px;
        display:inline-block;
        border: 1.5px solid white;
        border-radius:50%;
        margin: 0 5px;
    }

    #map-container {
        position:fixed;
        top:0;
        right:0;
        display: block;
        height:100vh;
          max-height: -webkit-fill-available;
        width: calc(100% - """+sidebar_width+""");
        transition:.15s;
    }

    iframe {
        position:fixed;
        border:none;
        right:0;
        top:0;
        height:100%;
        width: calc(100% - """+sidebar_width+""")
    }

    #search-container {
        height: auto;
        margin:0 0 10px;
        width:100%;
        padding:0;
        transition: .2s ease-in-out;    
        }


    #search-list{
        width: 100%
    }

    #searchbar {
            background-color:#424242;
            width: calc(100%) !important; 
            border-radius: 5px;
            border: 1.5px solid #448AFF !important;
            padding:10px;
            margin:0;
            font-size:18px;
            color:white;
        } 


    ::placeholder {
            font-size:18px;
            font-weight: bold;
            color:#82B1FF;
            opacity: 1 !important; 
            margin:0
        }

    #search-message-container{
    max-width:265;
    }

    #search-list{ 
        display:block;
        width: 100%;
        margin:0;
        font-size: 16px;
        max-height:250px;
        list-style-type: none;
        padding: 0 15px;
        overflow:auto;
        transition: .2s ease-in-out;
       }

    .search-items { 
       display: none;
       padding:12px 0 12px 8px;
       cursor: pointer;
       border-top: 1px solid white;
       transition:.2s;
      } 

    .search-items:hover {
        background-color: #424242 !important;
    }

    .search-items p {
        margin-bottom:2px
    }

    #search-prompt {
        display: block;
        overflow: hidden;
        max-height: 150px;
        margin: 0px 0px 5px;
        max-height: 150px;
        
        transition: .2s;
        font-size: 0.95em;
      }

    #search-instruct {
        overflow: hidden;
        max-height: 0;
        transition: .2s;
        padding: 0 8px;
        margin:0;

      }
    #search-alert {
        overflow: hidden;
        max-height: 0;
        transition: .2s;
        padding: 0 8px;
        margin:0;

      }
    #search-init {
        overflow: hidden;
        max-height: 0;
        transition: .2s;
        padding: 0 8px;
        margin:0;
      }


    .isd-indicator-search {
        display:block;
        position: relative;
        top:0;
        right:10px;
        float:right;
        color:#E0E0E0;
    }



    #how-collapse-btn {
      background-color: transparent;
      color: white;
      cursor: pointer;
      padding: 8px 0;
      width: 100%;
      border: none;
      text-align: left;
      outline: none;
      font-size: 15px;  
    }

    #how-collapsed {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.2s ease-out;
    }
    
    #license-collapse-btn {
      background-color: transparent;
      color: white;
      cursor: pointer;
      padding: 12px 0;
      width: 100%;
      border: none;
      text-align: left;
      outline: none;
      font-size: 15px;  
    }

    #license-collapsed {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.2s ease-out;
    }
    
    #listings-collapse-btn {
      background-color: transparent;
      color: white;
      cursor: pointer;
      padding: 12px 0;
      width: 100%;
      border: none;
      text-align: left;
      outline: none;
      font-size: 15px;  
    }

    #listings-collapsed {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.2s ease-out;
    }
    
    
    .centroid-list-item {
        background-color: #424242;
        margin: 10px 0;
        border-radius:8px;
        padding:10px;
        cursor:pointer;
        transition:.2s;
        border: 1.5px solid #424242;

    }
    
    .centroid-list-item:hover {
        border-color: #448AFF;
        background-color: #333333;
    }
    
    .centroid-list-item b {
        color: #448AFF;
        margin-bottom:2px;
    }
    
    .centroid-list-item p {
        font-size: .98em;
        margin-bottom:2px;
    }
    .centroid-list-item a {
        font-size: .96em;
        text-decoration:none !important;
    }
    
    .centroid-list-item a:hover {
    }
    
    .high-conf-btn {
        background-color: #424242;
        margin: 15px 0;
        border-radius:8px;
        padding:8px 15px;
        cursor:pointer;
        transition:.15s;
        border: 1.5px solid #424242;
    }
    
    .high-conf-btn:hover {
        background-color: #448AFF;
    }
    .high-conf-btn a:hover {
        color: white !important;
    }

    

    .host-info-link {
        color:#C5B358;
    }
    
    
    .host-info-link:hover {
        color:#a3923e;
    }
    
    .centroid-list-item a {
        text-decoration:none !important;
        }
        
        
    .license-claim-container ul {
        padding-left:18px; 
        list-style: disc;
        color: white !important;
    }

    
    .host-license-warning ul {
        padding-left:18px; 
        list-style: disc;
        font-size:14px !important;
        color: white !important 
    }
    
    .host-license-warning ul li::before {
      content: "\2022";
      color: white; 
      font-weight: bold; 
    }
    
    
    
    .modal {
      display: none; /* Hidden by default */
      position: fixed; /* Stay in place */
      z-index: 10000; /* Sit on top */
      left: 0;
      top: 0;
      width: 100vw; /* Full width */
      height: 100vh; /* Full height */
        min-height: -webkit-fill-available;

      padding:40px;
      overflow: auto; /* Enable scroll if needed */
      background-color: rgb(0,0,0); /* Fallback color */
      background-color: rgba(0,0,0,0.9); /* Black w/ opacity */
    }

    /* Modal Content/Box */
    .modal-content {
      background-color: #2b2b2b !important;
      z-index: 10000; /* Sit on top */
      padding: 0;
      height: calc(100vh - 80px);
        max-height: -webkit-fill-available;

      width: calc(100vw - 80px);
    }
    .modal-inner {
      padding: 40px;
      box-shadow: inset -12px -12px 10px rgba(0,0,0,0.25);
      z-index: 100001; /* Sit on top */
      overflow:auto;
      width: 100%;
      height: 100%;
      background-color:#2b2b2b !important;
    }

    /* The Close Button */
    .close {
      position:fixed;
      color: white !important;
      float: right;
      font-size: 30px;
      font-weight: bold;
      opacity: 1 !important;
      margin-top: -15px;
      
    }

    .close:hover,
    .close:focus {
      color: black;
      text-decoration: none;
      cursor: pointer;
    }

    .dataframe {
        border-color: #d4d4d4;
    }

    .dataframe td {
        white-space: pre;
        max-width:120px;
        padding: 3px;
        overflow:auto;
        color:white !important;
    }

    .dataframe th {
        max-width:120px;
        padding: 3px;
        color:white !important;
    }

    """


# JAVASCRIPT
    
    script = """
        document.getElementById("how-collapse-btn"
            ).addEventListener("click", function() {
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
              content.style.maxHeight = null;
              this.innerHTML = "[ + ] Learn more... ";
            } else {
              content.style.maxHeight = content.scrollHeight + "px";
              this.innerHTML = "[ &ndash; ] Learn less... ";
            } 
        });
    </script>
    <script>
    
    
    var viewportHeight = $('.banner').outerHeight();
    $('.banner').css({ height: viewportHeight });
    
    
    if ($('body').hasClass('mobile')) {
      var viewportHeight = $('.banner').outerHeight();
      $('.banner').css({ height: viewportHeight });
    }


    </script>
    <script>
    
        var modal = document.getElementById("myModal");

        // Get the button that opens the modal
        var btn = document.getElementById("myBtn");

        // Get the <span> element that closes the modal
        var span = document.getElementsByClassName("close")[0];

        // When the user clicks on the button, open the modal
        btn.onclick = function() {
          modal.style.display = "block";
        }

        // When the user clicks on <span> (x), close the modal
        span.onclick = function() {
          modal.style.display = "none";
        }

        // When the user clicks anywhere outside of the modal, close it
        window.onclick = function(event) {
          if (event.target == modal) {
            modal.style.display = "none";
          }
        }

    </script>
    <script>
    
    
        document.getElementById("mobile-switcher"
            ).addEventListener("click", switch_mobile_menu);
        document.getElementById("mobile-switch-link"
            ).addEventListener("click", switch_mobile_menu);
        document.getElementById("mobile-menu-circle"
            ).addEventListener("click", switch_mobile_menu);
            
        var mapOpen = false
        function switch_mobile_menu() {

            var navbar = document.getElementById("nav-banner");
            var sidebar = document.getElementById("sidebar-container");
            var map = document.getElementById("map-container");
            var slabel = document.getElementById("mobile-switch-label");
            var openico = document.getElementById("open-menu-icon");
            var slink = document.getElementById("mobile-switch-link");
            var menucir = document.getElementById("mobile-menu-circle");


            if ( mapOpen ){
                  sidebar.setAttribute("style","left:0") ;

                  map.setAttribute("style","left:calc(100vw - 100px)") ;
                  map.setAttribute("style","width:800px") ;

                  slabel.innerHTML = "<b>Map </b> <i class='fa fa-map' aria-hidden='true'></i>";
                  
                  openico.setAttribute("style","visibility:hidden") ;
                  menucir.setAttribute("style","left:-100px") ;
                  
                  slink.setAttribute("style","right:0") ;
                  slink.setAttribute("style","width:30px") ;
                  

                  mapOpen = false;
            }
            else {
                  sidebar.setAttribute("style","left:calc(-100vw + 40px)") ; 

                  map.setAttribute("style","left:10px !important") ;
                  map.setAttribute("style","width:calc(100vw - 10px) !important") ;

                  slabel.innerHTML = "<b>Details </b> <i class='fa fa-bars' aria-hidden='true'></i>";
                  
                  openico.setAttribute("style","visibility:visible") ;
                  menucir.setAttribute("style","left:-30px") ;

                  slink.setAttribute("style","width:20px") ;
                  slink.setAttribute("style","left:0px") ;

                  mapOpen = true;
            } 
        };

    
    </script>
    <script>
    
        document.getElementById("license-collapse-btn"
            ).addEventListener("click", function() {
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
              content.style.maxHeight = null;
              this.innerHTML = "[ + ] Show license details";
            } else {
              content.style.maxHeight = content.scrollHeight + "px";
              this.innerHTML = "[ &ndash; ] Hide license details";
              setTimeout( function (){
                      location.hash = "#licenses";
                            }, 250);
            } 
            
        });
    </script>
    <script>
    
        document.getElementById("listings-collapse-btn"
            ).addEventListener("click", function() {
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
              content.style.maxHeight = null;
              this.innerHTML = "[ + ] Show listings";
            } else {
              content.style.maxHeight = content.scrollHeight + "px";
              this.innerHTML = "[ &ndash; ] Hide listings";              
              setTimeout( function (){
                      location.hash = "#listings";
                            }, 250);
            } 
        });
    </script>
    <script>
    
        
    function search_res(container, value) {
    
        var item = container.toLowerCase().replace(/\s+/g, '');                
        var val = value.toLowerCase().replace(/\s+/g, '');
        var returnValue = false;
        var pos = item.indexOf(val);
        if (pos >= 0) {
            returnValue = true;
        }
        return returnValue;
    }
    
    </script>
    <script>
    
    document.getElementById('searchbar').addEventListener("click",
        function(){
                initOpenState = "True"
                let x = document.getElementsByClassName('search-items');
                var counter = 0
                for (i = 0; i < x.length; i++) { 
                    var dt = x[i].getAttribute("data-type");
                    if (dt=="cluster") { 
                        counter = counter + 1
                        x[i].style.display="list-item"; 
                    } 
                    if (counter == 3) {
                         break;
                    }
                } 
                document.getElementById('search-list').style.maxHeight = "400px";
                document.getElementById('search-list').style.padding = "0 15px 15px";
                let msg_init = document.getElementById('search-init'); 
                msg_init.style.maxHeight = "150px";
                msg_init.style.margin = "8px";
                
        }
    )
    
    
    document.getElementById('searchbar').addEventListener("focusout",
        function(){
            let input = document.getElementById('searchbar').value;
            if (input=="" && initOpenState !== "True"){
                document.getElementById('search-list').style.maxHeight = "0";
                document.getElementById('search-list').style.padding = "0 15px";
            
                let x = document.getElementsByClassName('search-items');
                for (i = 0; i < x.length; i++) {  
                    x[i].style.display="none"; 
                } 
                let msg_init = document.getElementById('search-init'); 
                msg_init.style.maxHeight = "0";
                msg_init.style.margin = "0";
            }

        }
    )

    
    function search_item() { 
        initOpenState = "False";
        
        let sb = document.getElementById('searchbar');
        let input = document.getElementById('searchbar').value;
        let searchList = document.getElementById('search-list');
        searchList.scrollTop = 0;

        let x = document.getElementsByClassName('search-items'); 
        
        let instruct = document.getElementById('search-instruct'); 
        let alert = document.getElementById('search-alert'); 
        let s_init = document.getElementById('search-init'); 
        
        let msg_init = document.getElementById('search-init'); 
        msg_init.style.maxHeight = "0";
        
        var visible = 0
        
        if (sb !== null && sb.value === "") {

            var counter = 0
            for (i = 0; i < 10; i++) {  
                    var dt = x[i].getAttribute("data-type");
                    if (dt=="cluster") { 
                        counter = counter + 1
                        x[i].style.display="list-item"; 
                    } 
                    if (counter == 3) {
                         break;
                    }
            } 
        }
        
        else {
            for (i = 0; i < x.length; i++) {
            
                if ( search_res(x[i].getAttribute("search-data"), input) ) { 
                    x[i].style.display="list-item"; 
                    visible += 1
                } 
                else { 
                    x[i].style.display="none";
                } 
             } 
        }

        if (sb.value !== "") {
                if (visible==0) {
                    instruct.style.maxHeight = "0";
                    instruct.transition = ".2";
                    instruct.style.margin = "0";
                    s_init.style.maxHeight = "0";
                    s_init.transition = ".2s";
                    s_init.style.margin = "0";
                    
                    alert.style.maxHeight = "150px";
                    alert.style.margin = "8px";
                   
                    
                    searchList.style.padding = "0";
                    searchList.style.maxHeight = "0";
                    

                    }
                else {
                    instruct.innerHTML = "<b>" + visible.toString() + "</b> results" ;
                    instruct.style.maxHeight = "150px";
                    instruct.transition = "0";
                    instruct.style.margin = "8px";
                    
                    
                    s_init.style.maxHeight = "0";
                    s_init.transition = ".2s";
                    s_init.style.margin = "0";
                    alert.style.maxHeight = "0";
                    alert.transition = ".2s";
                    alert.style.margin = "0";
                    
                    searchList.style.padding = "0 15px 15px ";
                    searchList.style.maxHeight = "305px";                    
                    }
 
            
          }         
        else {
            for (i = 0; i < x.length; i++) {  
                 x[i].style.display="none";
            } 
            instruct.style.maxHeight = "0";
            instruct.transition = ".2s";
            instruct.style.margin = "0"
            alert.style.maxHeight = "0";
            alert.transition = ".2s";
            alert.style.margin = "0";
            s_init.style.maxHeight = "0";
            s_init.style.margin = "0";
            searchList.style.maxHeight = "0";
            searchList.style.padding = "0";
                    }       
        }

    """
    
    return style, script 


# GENERATE HTML FOR THE HOME PAGE, GIVEN A MAP OBJECT AND ASSOCIATED DATA
def make_home_page(m, listings):
    
    style, script = gen_assets()

    # HOME PAGE SIDEBAR LAYOUT
    
    sidebar = f"""
    
        {config.search_html}
                   
        <hr>


        <h2>STR Violations Locator</h2>
        
        <p style='margin:10px 0;'>
        Viewing all <b>{len(listings)}</b> listings (hover/click for details). Use the search bar
        or browse the map to find specific hosts, listings and clusters.
        </p>
        

    
        <a href='/clusters.html'>
            <div class='high-conf-btn' onclick="location.href='/clusters.html'">
                <h3 style='color:white; margin:0'>
                View high-confidence clusters&#8594;</a>
                </h3>
                <p style='margin: 5px 0 0'>For pinpointing STR building locations</p>
            </div>
        </a>      
        
       <p>
        Airbnb randomly anonymizes listing coordinates by
        up to 500 feet. To determine actual locations of high-occupancy short-term
        rental buildings, listings are algorithmically grouped into
        suspected building clusters based on location, host and license data.
        Choose a listing from the map or <a href='/clusters.html'>view high-confidence clusters</a> (those with
        {config.min_top_centroid} or more listings) to get started.
        </p>
   
        <hr class='hr-lite'>
        
        <h3>Citywide Registration Rate</h3>
           <img style='width: calc(100% + 40px); margin-left: -30px;' src='status_pie.png'>
           
        <p class='text-lite' style='font-size:.95em'>
            <b>Note:</b> "Active (limit exceeded)" indicates the number of
            listings or accommodations exceeds the limit allowed per
            license type (Home Share, Owner-Occupied, or Owner-Adjacent), assuming conversatively
            that one listing equals one bed and one "accommodation" equals one guest.
        </p>
        

        <hr class='hr-lite'>

        <h3>License Status Details</h3>
        <p class='text-lite' style='font-size:.97em; margin-bottom:0'>
             Cross-referenced with offical <a target='blank' href='https://www.boston.gov/departments/inspectional-services'>ISD</a> data
        </p>

        {make_status_legend(listings, page_type='home')}
        
        

        <div style='margin-top:18px; font-size:.92em;'>
        
            <p style='margin-bottom:5px;'><b>Listings data last updated: {config.latest_scraped}</b></p>
            <p style='margin-bottom:5px;'><b>ISD data last updated: {config.ISD_updated}</b></p>

            
            <p class='text-lite'>
            <b>Data Sources:</b> InsideAirbnb; City of Boston
            Inspectional Services Department (ISD);
            Alliance of Downtown Civic Associations (ADCO).</p>
        </div>


    """


    # HOME PAGE PAGE HTML

    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>STR Monitor - Home</title>


    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    </head>
    <body>
    


        <div id='page-container'>
            <div id='sidebar-container'>
                    
                    <div id='nav-banner'>
                        <p>
                        <span style='cursor:pointer;'>
                            Boston Airbnb Monitor - Home
                        </span>
                        </p>
                        
                        <div id='mobile-switch-link'></div>
                        <div id='mobile-switcher'>
                            <p id='mobile-switch-label'>
                            <b>Map </b><i class="fa fa-map" aria-hidden="true"></i> 
                            </p>
                        </div>
                        
                        <div id='open-menu-icon'>
                            <span id="mobile-menu-circle">
                                <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            </span>
                        </div>
                    
                    </div>                    
                    
                <div id='sidebar'>
                    {sidebar}            
                </div>
                
            </div>

            <div id='map-container'>
                {m.get_root().render()}
            </div>

        </div>
        
        <script>
            {script}
        </script>

    </body>

    </html>
    """

    
    return page



# GENERATE HTML FOR HIGH_CONF CENTROIDS PAGE, GIVEN A MAP OBJECT AND ASSOCIATED DATA

def make_top_centroids_page(m, centroids, listings,):

    style, script = gen_assets()

    # HOME PAGE SIDEBAR LAYOUT

    sidebar = f"""

        
        {config.search_html}

        <hr>

        <h2> High-confidence clusters </h2>

        <p style='color:white'>
        Airbnb listings are clustered algorithmically based on host, license and semi-anonymous geographic data.
        The goal is to identify groups of listings that likely belong to one building.
        </p>
        
        <a id="how-collapse-btn">[+] Learn more... </a>
        <div style='font-size.9em;color:white;' id="how-collapsed">
            <p style='margin-top:5px'>
            Dbscan clustering algorithms are applied over multiple phases to acquire final
            best-guess locations for suspected high-occupancy STR buildings. Listings are 
            are split into unique host-license groups for the first phase of clustering,
            overestimating cluster granularity, to generate many small groups.
            </p><p>
            In the second phase, dbscan with a lower distance threshhold combines nearby
            sub-groups owned by the same host or corporation group. This acccounts for cases
            where a host uses multiple license numbers within the same building, which would
            otherwise be clustered separetely.
            </p><p>
            In buildings with many assumed listings, we can then exploit the semi-anonymized public
            location data to approximate a guess for the actual building location by 
            taking the average coordinates of all listings in a cluster. This strategy
            often leads directly to or near a building with confirmed STR units inside.
            </p><p>
            A repository containing the complete stack of analysis is available on GitHub:
            
            <a href='https://github.com/js-fitz/airbnb_tracker'><b>Github Repo</b></a>
            
            </p>
        </div>



        <hr class='hr-lite'>

        <h3>License Status Summary</h3>
        <p class='text-lite' style='font-size:.97em; margin-bottom:0'>
             Cross-referenced with offical <a target='blank' href='https://www.boston.gov/departments/inspectional-services'>ISD</a> data
        </p>

        {make_status_legend(listings[listings.high_confidence==True], True)}

        <div style='margin-top:18px'>
            <p class='text-lite' style='font-size:.92em;'>
            <b>Note:</b> This page includes data only for hosts
            with at least one suspected "super-building", i.e.
            {config.min_top_centroid} or more listings in a single cluster. For smaller clusters and more listings,
            use the search bar or <a href='/'>return to the home page</a>.</p>
        </div>


    """


    # HOME PAGE PAGE HTML

    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>High-Confidence Clusters</title>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    </head>
    <body>

        <div id='page-container'>
            <div id='sidebar-container'>
                    
                    <div id='nav-banner'>
                        <p>
                            <span style='cursor:pointer;'>
                                <a style='color:#448AFF' href='/'>Home</a> &#62; Clusters
                            </span>
                        </p>
                        
                        <div id='mobile-switch-link'></div>
                        <div id='mobile-switcher'>
                            <p id='mobile-switch-label'>
                            <b>Map </b><i class="fa fa-map" aria-hidden="true"></i> 
                            </p>
                        </div>
                        
                        <div id='open-menu-icon'>
                            <span id="mobile-menu-circle">
                                <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            </span>
                        </div>
                    
                    </div>
                    
                    
                    
                <div id='sidebar'>
                    {sidebar}            
                </div>
                
            </div>

            <div id='map-container'>
                {m.get_root().render()}
            </div>

        </div>
        
        <script>
            {script}
        </script>

    </body>

    </html>
    """

    
    return page


# GENERATE MODAL FOR LISTINGS DETAILS (tabulated group data)

def get_listing_modal(listings, group_type, idx):
    
    root_dir = 'airbnb_tracker'
    save_dir = f'/{group_type}/{idx}—data.csv'
    out = listings.copy()
    out = out.rename(columns={'centroid_group':'cluster'})
    out.drop(columns=['amenities'], inplace=True)
    
    
    l = listings.copy()
    l = l[['id', 'listing_url',  'centroid_group',  'name', 'host_id',
            'host_url', 'host', 'neighbourhood_cleansed', 'latitude',
            'longitude', 'property_type', 'room_type', 'accommodates',
            'bedrooms', 'beds', 'price', 'minimum_minimum_nights',
            'availability_365', 'number_of_reviews', 'license',  'host_alias',
            'alias_id', 'status', 'ISD_status', 'ISD_address', 'ISD_category',
            'license_listing_count', 'license_cat_max_exceed', 'simple_status',
            'full_status', 'centroid_group_distance', 'last_scraped',]]
    l = l.rename(columns={'centroid_group':'cluster'})
    for c in l.columns:
        if len(c)>10:
            l = l.rename(columns={c:c.replace('_', '<br>').capitalize()})
        else:
            l = l.rename(columns={c:c.replace('_', ' ').capitalize()})

    table = l.to_html(index=False, render_links=True, escape=False, show_dimensions=True)
    
    if 'host' in group_type:
        header = f"Host: <b>{listings.host.values[0]}</b>"
    else:
        header = f"Cluster: <b>{idx}</b>"
    
    out = f"""
    
    <button id="myBtn" style='margin-top:px'>Detailed listings data
        <i class="fa fa-external-link-square" aria-hidden="true"></i>
    </button>

    <div id="myModal" class="modal">

      <!-- Modal content -->
      <div class="modal-content">
      <div class="modal-inner">
        <span class="close">&times;</span>
        
        
        <h2 style='margin-top:15px;'>
            {header}
        <h2>
        <h3>
        <span> {len(listings)} total listings </span> | 
        
        <a href='{save_dir}' download="{group_type}-{idx}_listings.csv">
                            <i class="fa fa-download" aria-hidden="true"></i>
                            Download data as a CSV </a>
        </h3>
        <p>
        {table}
        </p>  
      </div>
      </div>

    </div>
    
    """
    
    return out



# GENERATE HTML LIST OF LISTINGS (at the bottom of the sidebar on a centroid or host page)


def get_listing_list(listings, group_type, idx):
    
    root_dir = 'airbnb_tracker'
    save_dir = f'/{group_type}/{idx}—data.csv'
    out = listings.copy()
    out = out.rename(columns={'centroid_group':'cluster'})
    out.drop(columns=['amenities'], inplace=True)
    out.to_csv(root_dir+save_dir, index=False)
    
    listing_list = f"""<p style='font-size:.9em;text-align:right'>
                        <a href='{save_dir}' download="{group_type}-{idx}_listings.csv">
                            <i class="fa fa-download" aria-hidden="true"></i>
                            Download {group_type} data as a CSV </a>"""
    is_multihost = listings.alias_id.nunique()>1
    
    for row in listings.index:
        
        r_data = listings.loc[row]

        row_html = f"""
        <a href='{r_data.listing_url}' target='_blank'>
        <div class='centroid-list-item'>
            
                <p style='font-size:1em'><b>{r_data['id']}</b></p>
                <p style='font-size:1.1em'>{r_data['name']}</p>
            """

        if is_multihost:
            row_html +=  f"""
                <p><span class='text-lite'>Alias account:</span>
                    {r_data.host_alias}</p> """
        
        min_nights = str(r_data.minimum_minimum_nights).replace('.0', '')
        row_html += f"""
                <p> <span class='text-lite'>Minimum nights per stay:</span>
                    <b>{min_nights}</b></p>
                <p> <span class='text-lite'>License:</span>
                    <b>{r_data.license}</b> ({r_data.status})</p>
                    
                <a href='{r_data.listing_url}' target='_blank'>View listing photos &#8594;</a>
            
        </div>
        </a>

        """        
        listing_list += row_html
    return listing_list


# GENERATE HTML FOR THE A HOST PAGE, GIVEN A MAP OBJECT AND ASSOCIATED DATA

def make_host_focus_page(feature, group, m, centroids, listings, sub_centroids):
    
    style, script = gen_assets()
    centroids = centroids[centroids[feature]==group].copy()
    listings = listings[listings[feature]==group].copy()
    sub_centroids = sub_centroids[sub_centroids[feature]==group].copy()

    h_name = listings.host.values[0]
    h_id = listings.host_id.values[0]

    is_corpgroup = 'Corp' in h_id
        
    centroids['subnums'] = centroids.centroid_group.apply(lambda x: int(x.split('-')[-1].replace('s','')) )
    
    cluster_list = ''
    for c_group in centroids.sort_values('subnums').centroid_group.values:
        c_data = centroids[centroids.centroid_group==c_group]
        l_data = listings[listings.centroid_group==c_group]
        
        row_html = f"""
        <a href='/cluster/{c_group}.html'>
        <div class='centroid-list-item'>
            <p style='font-size:1.2em'><b>{c_group}</b> ({len(l_data)} listings)</p>
            <p><span class='text-lite'>At or near:</span> {c_data.best_address.values[0].split('Boston')[0]} </p>
            <a href='/cluster/{c_group}.html'>View cluster listings & details &#8594;</a>
        </div>
        </a>

        """        
        cluster_list += row_html
    
    # HOME PAGE SIDEBAR LAYOUT

    if len(h_name)<15: title_size= 2.2
    elif len(h_name)<25: title_size= 2
    else: title_size = 1.8
    
    sidebar = f"""

        <h2 style='margin: 0 0 10px; font-size:{title_size}em;'> {h_name} </h2>
        <h3>
            <b>{len(listings)} listing{'s' if len(listings)!=1 else ''}</b> | 
            {len(centroids)} cluster{'s' if len(centroids)!=1 else ''}
        </h3> """
        
    if is_corpgroup:
        sidebar += """
        <p class='text-lite'> <b>Note:</b> This host represents a corporation using the following 
             alias accounts</p>
             <ul style='margin:0; font-size:15px; color: white; padding-left:15px;margin-left:10px' >
        """
        for alias in listings.alias_id.dropna().unique().astype(int):
            alias_data = listings[listings.alias_id==alias]
            a_name = alias_data.host_alias.values[0]
            alias = int(alias)
            sidebar += f"""<li>
            <a target='_blank' href='https://www.airbnb.com/users/show/{alias}'>
                "{a_name}"</a> - {len(alias_data)} listings</li>"""

        sidebar += "</ul>"
        
    
    shared_licenses = listings[listings.other_hosts.notna()].copy()
    not_shared = (len(listings)-len(shared_licenses)) >0
    if len(shared_licenses):
        sidebar += f"""
        
        <table class='text-lite' style='margin-top:8px' >
            <tr>
            <td style='vertical-align:top'>
                <i class="fa fa-clone" style='color:#EF5350; font-size:1.3em; margin: 5px 10px 0 0'></i>
            </td>
            <td>
            <p>
                In {'' if not_shared else 'all '}{len(shared_licenses)} listing{'s' if len(shared_licenses)!=1 else ''}
                this host uses a license number also in use by another host:
            </p>
            </td>
        </table>
        
            
            <ul class='host-license-warning'>"""
        
        for license in shared_licenses.license.unique():
            lic_data = shared_licenses[shared_licenses.license==license]
            href = f"/cluster/{lic_data.centroid_group.values[0]}.html"
            sidebar += f"""<li>
                        <p>
                        <a href='{href}'>{license} (used by <b>{shared_licenses.host_id.nunique()}</b> other 
                        host{'s' if  shared_licenses.host_id.nunique()!=1 else ''})</a>
                           </p></li>"""
            
        sidebar += '</ul>'
    
    
    sidebar+= f"""
    
            {get_listing_modal(listings, 'host', h_id)}

    
        <hr class='hr-lite'>

        <h3>Host License Details</h3>
        <p class='text-lite' style='font-size:.97em; margin-bottom:0'>
             Cross-referenced with offical <a target='blank' href='https://www.boston.gov/departments/inspectional-services'>ISD</a> data
        </p>

        {make_status_legend(listings)}
        
        <hr class='hr-lite' id='clusters'>

        <h3 style='margin-bottom:18px'>Host clusters</h3>
        
        {cluster_list}
        
        <hr class='hr-lite' id='listings'>
                                
        <a id="listings-collapse-btn" style='margin:15px 0'>
            [ + ] Show listings
        </a>

        <div id='listings-collapsed'>

            <h3 style='margin:18px 0 8px'>Listings</h3>
            <p class='text-lite' style='font-size:13px; margin: 2px 0 12px'>
                <b>Note:</b> If a listing link redirects to the Airbnb.com
                homepage, it indicates the host has removed the listing
                since the last update from <a href='http://insideairbnb.com/'>InsideAirbnb</a>.</p>
            {get_listing_list(listings, 'host', h_id)}

        </div>



    """


    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>{h_name} - Host Overview</title>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    </head>
    <body>

        <div id='page-container'>
            <div id='sidebar-container'>
                    
                    <div id='nav-banner'>
                        <p>
                            <a href='/'>Home</a> &#62;
                            Hosts &#62;
                            <span style='color:#448AFF'>{h_name}</span>
                        </p>
                        
                        <div id='mobile-switch-link'></div>
                        <div id='mobile-switcher'>
                            <p id='mobile-switch-label'>
                            <b>Map </b><i class="fa fa-map" aria-hidden="true"></i> 
                            </p>
                        </div>
                        
                        <div id='open-menu-icon'>
                            <span id="mobile-menu-circle">
                                <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            </span>
                        </div>
                        
                    </div>
                    
                <div id='sidebar'>
                    {sidebar}            
                </div>
                
            </div>

            <div id='map-container'>
                {m.get_root().render()}
            </div>

        </div>
        
        <script>
            {script}
        </script>

    </body>

    </html>
    """

    
    return page


# GENERATE HTML FOR A CENTROID PAGE, GIVEN A MAP OBJECT AND ASSOCIATED DATA

def make_centroid_page(feature, group, m, listings, centroids, sub_centroids):
    
    style, script = gen_assets('cluster')
    
        
    centroids = centroids[centroids[feature]==group].copy()
    listings = listings[listings[feature]==group].copy()
    sub_centroids = sub_centroids[sub_centroids[feature]==group].copy()
    c_data = centroids.reset_index()
    c_data = c_data.loc[0] # convert from df to row object
    
    c_group = c_data.centroid_group
    h_name = c_data.host
    h_id = c_data.host_id
    
    
    
    # HOME PAGE SIDEBAR LAYOUT

    sidebar = f"""
        <h2 style='font-size:2.2em; margin:0 0 10px'> Cluster {c_group} </h2>

        <h3 style='margin-bottom:5px'>
            <a href='#listings'><b>{len(listings)} listing{'s' if len(listings)!=1 else ''}</b></a> |
            {max(1, listings.alias_id.nunique())} account{'s' if listings.alias_id.nunique()!=1 else ''}</b> 
        </h3> 
        
        <p class='text-lite'>
            <b>{listings.accommodates.sum()}</b> total accommodations
        </p> 
        <hr class='hr-lite'>
        <h3 style='font-size: 1.6em; margin:15px 0 5px'>
        Host: <a href='/host/{h_id}.html' class='host-info-link'> {h_name}</a> </h3>
        """
    if len(re.findall('\d', str(list(c_data.license_dict.keys())[0]) )):
        sidebar += f"""
        <p class='text-lite;margin-top:10px;margin-bottom:3px'>
            License <b>{list(c_data.license_dict.keys())[0]}</b> status: </p>
            {get_status_indicator_html(c_data, size='1.2em', text=True, page_info=True)}
        <div style='margin-bottom:8px'></div>

        """
 

               
    if listings.alias_id.nunique()>1: # if corporate group AKA multiple hosts
        sidebar += """
        <p class='text-lite'><b>Note:</b> This host represents a corporation using multiple alias accounts at this location</p>
             <ul style='margin:0; line-height:22px; font-size:16px; color: white; padding-left:15px;margin-left:10px' >
        """
        for alias in listings.alias_id.dropna().unique().astype(int):
            alias_data = listings[listings.alias_id==alias]
            a_name = alias_data.host_alias.values[0]
            alias = int(alias)
            sidebar += f"""<li>
            "<a target='_blank' href='https://www.airbnb.com/users/show/{alias}'>
                {a_name}</a>" ({len(alias_data)} listings)</li>"""
        sidebar += "</ul>"
        
        
    # make shared licenses list
    shared_licenses = listings[listings.other_hosts.notna()].copy()
    not_shared = (len(listings)-len(shared_licenses)) >0
    if len(shared_licenses):
        sidebar += f"""
        <table class='text-lite' style='margin-top:8px' >
            <tr>
            <td style='vertical-align:top'>
                <i class="fa fa-clone" style='color:#EF5350; font-size:1.3em; margin: 5px 10px 0 0'></i>
            </td>
            <td>
            <p>
                {'' if not_shared else 'All '}{len(shared_licenses)}
                listing{'s in this cluster use' if len(shared_licenses)!=1 else 'in this cluster uses'}
                a license also in use by another host:
            </p>
            </td>
        </table>
            <ul class='host-license-warning'>"""
        for license in shared_licenses.license.unique():
            lic_data = shared_licenses[(shared_licenses.license==license)]
            other_host_id = lic_data.other_hosts.values[0]
            
            href = f"/host/{other_host_id}.html"
            sidebar += f"""<li>
                        <p>
                        <a href='{href}'>{license} (used by <b>{lic_data.host_id.nunique()}</b> other 
                        host{'s' if lic_data.host_id.nunique()!=1 else ''}) &#8594;</a>
                           </p></li>
            """
        sidebar += "</ul>"

        
        # address info:
    sidebar += f"""
        <p style='margin-top:10px; margin-bottom:8px'>{'At or near' if c_data.best_address_type=='GCODE' else 'Address'}:</p>
        """
    
    if config.sub_mode:
        c_sub_centroids = listings.sub_centroid.dropna().nunique()
        if c_sub_centroids: # if multilpe sub-centroids found
            sidebar += """
            <p class='text-lite' style='color:white!important;font-size:14px'>
            <b style='color:#448AFF'>Multiple sub-clusters detected - </b>
            choose a sub-cluster (in blue) from the map for details</p>
            """
        
    sidebar += f"""<p style='font-size:1.25em; margin-bottom:5px'><b>{c_data.best_address.split('Boston')[0]}</b></p>
        <p class='text-lite' style='font-size:.95em; margin:0'>
                        <b>Source:</b>
    """
    
    if c_data.best_address_type=='ISD':
            sidebar += f"""
                     Verified license claimed {c_data.ISD_match_type.lower()}</p>
                """
    if c_data.best_address_type=='GCODE':
            sidebar += f"""
                        Inferred centerpoint of cluster listings</p>
                    <p style='font-size:.98em'>
                        Estimated accuracy: <b style='color:#448AFF'>{c_data.confidence}</b></p>
                """
            

    
    
    sidebar+= f"""
    
        {get_listing_modal(listings, 'cluster', c_group)}
    
        <hr class='hr-lite'>
        
        <h3 style='margin-bottom:12px'>License Status Details</h3>
        
        {make_status_legend(listings)}
        
            <div id='licenses' class='license-claim-container'>
            <a id="license-collapse-btn" style='margin:15px 0 15px;'>
                [ + ] Show license details
            </a>

            <div id="license-collapsed">
                    <p style='margin-top:10px;>
                        {c_data.ISD_match_type if str(c_data.ISD_match_type)!='nan' else ''}</p>
                        {c_data.ISD_bldg_details}
            </div>
            </div>
        
        
        <hr class='hr-lite' id='listings'>
        
        <a id="listings-collapse-btn" style='margin:15px 0'>
            [ + ] Show listings
        </a>

        <div id='listings-collapsed'>

            <h3 style='margin:18px 0 8px'>Listings</h3>
            <p class='text-lite' style='font-size:14px; margin: 2px 0 120x'>
                <b>Note:</b> If a listing URL redirect to the Airbnb.com
                homepage, it suggests the host has removed the listing
                since the last update from <a href='http://insideairbnb.com/'>InsideAirbnb</a></p>
            {get_listing_list(listings, 'cluster', c_group)}

        </div>

    """


    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Cluster {c_group}</title>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    </head>
    <body>

        <div id='page-container'>
            <div id='sidebar-container'>
                    
                    <div id='nav-banner'>
                        <p>
                            <a href='/'>Home</a> &#62;
                            <a href='/clusters.html'>Clusters</a> &#62;
                            <span style='color:#448AFF'>{c_group}</span>
                            <a href='/host/{h_id}.html' style='color:#9C9C9C'>
                                ({h_name})</a>
                        </p>
                        
                        
                        <div id='mobile-switch-link'></div>
                        <div id='mobile-switcher'>
                            <p id='mobile-switch-label'>
                            <b>Map </b><i class="fa fa-map" aria-hidden="true"></i> 
                            </p>
                        </div>
                        
                        <div id='open-menu-icon'>
                            <span id="mobile-menu-circle">
                                <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            </span>
                        </div>
                        
                    </div>
                    
                <div id='sidebar'>
                    {sidebar}            
                </div>
                
            </div>

            <div id='map-container'>
                {m.get_root().render()}
            </div>

        </div>
        
        <script>
            {script}
        </script>

    </body>

    </html>
    """

    
    return page
        
    
# GENERATE HTML FOR A SUB-CENTROID PAGE, GIVEN A MAP OBJECT AND ASSOCIATED DATA


def make_sub_centroid_page(group, m, listings, sub_centroids):
    
    style, script = gen_assets()
    feature = 'sub_centroid'
        
    listings = listings[listings[feature]==group].copy()
    c_data = sub_centroids[sub_centroids[feature]==group].copy()
    parent_group = listings.centroid_group.values[0]
    
    c_group = c_data.centroid_group.values[0]
    h_name = c_data.host.values[0]
    h_id = c_data.host_id.values[0]
    
    
    # HOME PAGE SIDEBAR LAYOUT

    sidebar = f"""
        <p style='margin:0'> <a href='/cluster/{parent_group}.html'>
        &larr; Return to cluster <b>{parent_group}</b> </a> </p>

        <h2 style='font-size:2.2em; margin:0 0 10px'> (Sub)cluster {group} </h2>

        <h3 style='margin-bottom:5px'>
            <a href='#listings'><b>{len(listings)} listing{'s' if len(listings)!=1 else ''}</b></a> |
            {max(1, listings.alias_id.nunique())} account{'s' if listings.alias_id.nunique()!=1 else ''}</b> 
        </h3> 
        
        <p class='text-lite'>
            <b>{listings.accommodates.sum()}</b> total accommodations
        </p> 
        <hr class='hr-lite'>
        <h3 style='font-size: 1.6em; margin:15px 0 5px'>
        Host: <a href='/host/{h_id}.html' class='host-info-link'> {h_name}</a> </h3>
        """
    c_data = c_data.reset_index()
    c_data = c_data.loc[0]
    if len(c_data.license_dict.keys()):
        if len(re.findall('\d', str(list(c_data.license_dict.keys())[0]) )):
            sidebar += f"""
            <p class='text-lite;margin-top:10px;margin-bottom:3px'>
                License <b>{list(c_data.license_dict.keys())[0]}</b> status: </p>
                {get_status_indicator_html(c_data, size='1.2em', text=True, page_info=True)}
            <div style='margin-bottom:8px'></div>

            """
 

               
    if listings.alias_id.nunique()>1: # if corporate group AKA multiple hosts
        sidebar += """
        <p class='text-lite'><b>Note:</b> This host represents a corporation using multiple alias accounts at this location</p>
             <ul style='margin:0; line-height:22px; font-size:16px; color: white; padding-left:15px;margin-left:10px' >
        """
        for alias in listings.alias_id.dropna().unique().astype(int):
            alias_data = listings[listings.alias_id==alias]
            a_name = alias_data.host_alias.values[0]
            alias = int(alias)
            sidebar += f"""<li>
            "<a target='_blank' href='https://www.airbnb.com/users/show/{alias}'>
                {a_name}</a>" ({len(alias_data)} listings)</li>"""
        sidebar += "</ul>"
        
        
    # make shared licenses list
    shared_licenses = listings[listings.other_hosts.notna()].copy()
    not_shared = (len(listings)-len(shared_licenses)) >0
    if len(shared_licenses):
        sidebar += f"""
        <table class='text-lite' style='margin-top:8px' >
            <tr>
            <td style='vertical-align:top'>
                <i class="fa fa-clone" style='color:#EF5350; font-size:1.3em; margin: 5px 10px 0 0'></i>
            </td>
            <td>
            <p>
                {'' if not_shared else 'All '}{len(shared_licenses)}
                listing{'s in this cluster use' if len(shared_licenses)!=1 else 'in this cluster uses'}
                a license also in use by another host:
            </p>
            </td>
        </table>
            <ul class='host-license-warning'>"""
        for license in shared_licenses.license.unique():
            lic_data = shared_licenses[(shared_licenses.license==license)]
            other_host_id = lic_data.other_hosts.values[0]
            
            href = f"/host/{other_host_id}.html"
            sidebar += f"""<li>
                        <p>
                        <a href='{href}'>{license} (used by <b>{lic_data.host_id.nunique()}</b> other 
                        host{'s' if lic_data.host_id.nunique()!=1 else ''}) &#8594;</a>
                           </p></li>
            """
        sidebar += "</ul>"

        
        # address info:
    sidebar += f"""
        <p style='margin-top:10px; margin-bottom:8px'>{'At or near' if c_data.best_address_type=='GCODE' else 'Address'}:</p>
        """
    
    c_sub_centroids = listings.sub_centroid.dropna().nunique()
    if c_sub_centroids: # if corporate group AKA multiple hosts
        sidebar += """
        <p class='text-lite' style='color:white!important;font-size:13px'>
        <b style='color:#448AFF'>Multiple sub-clusters detected - </b>
        use the map to verify actual building locations. Central location:</p>
        """
        
    sidebar += f"""<p style='font-size:1.25em; margin-bottom:5px'><b>{c_data.best_address.split('Boston')[0]}</b></p>
        <p class='text-lite' style='font-size:.95em; margin:0'>
                        <b>Source:</b>
    """
    
    if c_data.best_address_type=='ISD':
            sidebar += f"""
                     Verified license claimed {c_data.ISD_match_type.lower()}</p>
                """
    if c_data.best_address_type=='GCODE':
            sidebar += f"""
                        Inferred centerpoint of cluster listings</p>
                    <p style='font-size:.98em'>
                        Estimated accuracy: <b style='color:#448AFF'>{c_data.confidence}</b></p>
                """
            

    
    
    sidebar+= f"""
        <hr class='hr-lite'>
        
        <h3 style='margin-bottom:12px'>License Status Details</h3>
        
        {make_status_legend(listings)}
        
        
        <hr class='hr-lite' id='listings'>
        
        <a id="listings-collapse-btn" style='margin:15px 0'>
            [ + ] Show listings
        </a>

        <div id='listings-collapsed'>

            <h3 style='margin:18px 0 8px'>Listings</h3>
            <p class='text-lite' style='font-size:14px; margin: 2px 0 120x'>
                <b>Note:</b> If a listing URL redirect to the Airbnb.com
                homepage, it suggests the host has removed the listing
                since the last update from <a href='http://insideairbnb.com/'>InsideAirbnb</a></p>
            {get_listing_list(listings, 'sub_cluster', group)}

        </div>

    """


    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Cluster {c_group}</title>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>{style}</style>

    </head>
    <body>

        <div id='page-container'>
            <div id='sidebar-container'>
                    
                    <div id='nav-banner'>
                        <p>
                            <a href='/'>Home</a> &#62;
                            <a href='/clusters.html'>Clusters</a> &#62;
                            <a href='/cluster/{parent_group}.html'>
                                {parent_group}</a> &#62;
                            <span style='color:#448AFF'>{group}</span>
                            <a href='/host/{h_id}.html' style='color:#9C9C9C'>
                                ({h_name})</a>
                        </p>
                        
                        
                        <div id='mobile-switch-link'></div>
                        <div id='mobile-switcher'>
                            <p id='mobile-switch-label'>
                            <b>Map </b><i class="fa fa-map" aria-hidden="true"></i> 
                            </p>
                        </div>
                        
                        <div id='open-menu-icon'>
                            <span id="mobile-menu-circle">
                                <i class="fa fa-arrow-right" aria-hidden="true"></i>
                            </span>
                        </div>
                        
                    </div>
                    
                <div id='sidebar'>
                    {sidebar}            
                </div>
                
            </div>

            <div id='map-container'>
                {m.get_root().render()}
            </div>

        </div>
        
        <script>
            {script}
        </script>

    </body>

    </html>
    """

    
    return page
        
