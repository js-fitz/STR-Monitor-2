# This mod contains functions to prep the data for mapping

import os
import re
import time
import config
import pickle
import folium
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.cluster import dbscan
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="••••••••••••")




# # # # # # #  PRE-CLUSTER DATA PREP   # # # # # # # # 


config.data_dir = 'data_raw/'
config.cache_dir = 'cache/'
config.save_dir = 'listings_clean/'



# Use ADCO internal decoder list to identify host "alias" accounts
# (multiple accounts used by the same corporation)
def id_corp_groups(listings):
    print(' ', '—'*30)
    
    # load & breifly clean decoder data
    decoder = pd.read_csv(f'{config.data_dir}STR Host Decoder - Sheet1.csv') # custom host alias decoder
    decoder.columns = decoder.loc[0].copy() # correct index
    decoder = decoder.loc[1:len(decoder)-2] # fix header row location
    decoder['Corp'] = decoder['Corp'].fillna(method='ffill')
    decoder = decoder[['Corp', 'host_id', 'host_name']] # drop unneeded columns
    decoder.host_id = decoder.host_id.astype(int) # numerify
    print('  > Searching listings data for alias host accounts...')
    
    # list matching account between the decoder & the actual airbnb data 
    found = [h_id for h_id in decoder.host_id.unique()
             if h_id in list(listings.host_id)]
    
    # compile the host account-corporpation identity map (alias account groups)
    host_group = {}
    group_idx = 0 # init
    for corp in decoder.Corp.unique(): # iterate known corporations
        corp_data = decoder[decoder.Corp==corp]
        hosts = list(corp_data.host_id.unique()) # list known alias accounts for this corporation 
        host_names = list(corp_data.host_name.unique())
        if len(hosts)>0:
            group_idx +=1 # iterate index to name this corporation corp_group_n
            corp = re.sub('[A-Z]- ?', '', corp) # remove extraneous characters
            h_idx = f'CorpGroup{group_idx}'
            
            # save the match corporation index data for this host account 
            for host in hosts:
                host_group[host] = {'host_name': corp,
                                    'host_id': h_idx,
                                    'group_hosts': ', '.join(host_names),
                                   }
    print(f'    > Found {len(found)}/{decoder.host_id.nunique()} active alias host accounts...')

    # reassign findings to listings data...
    for row in listings.index: # iterate all listings
        row_data = listings.loc[row]
        if row_data.host_id in host_group.keys(): # only affect listing with a matching corporation found:
            
            # save given host_name and host_id as "alias" versions:
            listings.loc[row, 'host_alias'] = listings.loc[row, 'host_name']
            listings.loc[row, 'alias_id'] = listings.loc[row, 'host_id']
            
            # overwrite given host_name and host_id with corporation group data:
            listings.loc[row, 'host_name'] = host_group[row_data.host_id]['host_name']
            listings.loc[row, 'host_id'] = host_group[row_data.host_id]['host_id']
    
    print(f"  >>> Generated {group_idx} new CorpGroup host IDs")
    return listings
    # done identifying corporate alias account groups.



# as named - loads raw listings data & parses license numbers
# (verifies licenses against ISD data)
def load_parsed_listings(sample_test=False): # for testing 
    
    # load either new or old scrape, depending on config setting
    if config.old: file = 'listings_old.csv'
    else: file = 'listings.csv'

    print('—'*60)
    print(f"Loading & parsing listings data")
    print('—'*60)
    listings = pd.read_csv(f'{config.data_dir}{file}')
    # random slice for efficiency if testing:
    if sample_test: listings = listings.sample(2500, random_state=3)
    print(f"  > '{file}' loaded")
    
    # for SEPTEMBER 2018: new data has a different price format
    if not config.old:
        listings.price = listings.price.apply(lambda x: float(''.join(x.replace(',', '').split('$')[1])))
    
    # convert host names and ids to corporation groups with function
    listings = id_corp_groups(listings)
    
    # Parse & verify license numbers...
    print(f"  > Parsing license numbers")
    
    # encode claimed licenses (using keyword indicators)
    def parse_str_num(l): # from listings data
        license = str(l).lower()
        if 'hospital' in license:
            return 'STRHS'
        if 'business' in license:
            return 'STRES'
        if 'hotel' in license:
            return 'STRLH/STRBB'
        if 'str' in license: # if numeric, use regex to return the license number
            return ''.join(re.findall('\d{6}', license))
        elif 'C0' in license:
            return license.upper().split('\n')[0]
       
        # probably either a number or missing:
        if license!='nan': # this would be a number without 'str' in front of it
            try: return ''.join(re.findall('\d{6}', license))
            except: pass # no number found, will result in nan
            
        else: return 'Missing' # none claimed  
    listings.license = listings.license.apply(parse_str_num)
    
    # log initial status
    print(f"    > Imported + licenses parsed")
    print(' ', '—'*30)
    print("  > Importing ISD registry data...")
    
    # load ISD data (official license registry from the city)
    isd = pd.read_csv('data_raw/ISD data.csv')
    def parse_license_num(l):
        return (''.join(re.findall('[\d]*', str(l).split('\n')[0]))) or 0
    isd['license_num'] = isd['License #'].apply(parse_license_num).astype(str)
    print("    > Imported + licenses parsed")
    
   
    # sort ISD data for most recent results (by indexing .values[0] inside the loop)
    # -- so if one license has multiple entries, we get the latest
    isd['issued'] = pd.to_datetime(isd['Issued Date'].fillna('2019-01-01'))
    isd.sort_values('issued', ascending=False)
    
    # define legal details for each license type category with standard LT abbreviations
    # based on the official city ordinance from boston.gov

    license_cat_dict = {
    'HS':{'name':'Home Share Unit',
          'maxbeds':5,
          'maxguests':10}, - # legal limit given license type
    'LS':{'name':'Limited Share Unit',
          'maxbeds':3,
          'maxguests':6},
    'OA':{'name':'Owner-Adjacent Unit',
          'maxbeds':5,
          'maxguests':10},
    'STRES':{'name':'Exempt: Executive Suite',
          'maxbeds':9999,
          'maxguests':999}, # 999 <= (unknown legal limit for exemptions)
    'STRLH':{'name':'Exempt: Lodging House',
          'maxbeds':999,
          'maxguests':999},
    'STRBB':{'name':'Exempt: Bed & Breakfast Suite',
          'maxbeds':999,
          'maxguests':999},
    'STRLH/STRBB':{'name':'Exempt: Lodging House / B&B'},
    'STRHS':{'name':'Exempt: Hospital Stays',
          'maxbeds':999,
          'maxguests':999},
    }
    
    # BEGIN LONG LOOP
    # for a given license claimed in the airbnb data, define the license status...
    print(' ', '—'*30)
    print('  > Parsing license statuses...')
    
    config.isd = isd # work with the global ISD data
    for license in tqdm(listings.license.unique()): # ITERATE EACH UNIQUE CLAIMED LICENSES
        
        listings_license_data = listings[listings.license==license].copy()
        full_status, simple_status = False, False # these MAY get overwritten during the loop
        
        # license NOT CLAIMED, EXEMPT OR MISSING
        if 'STR' in license or 'Missing' in license:
            ISD_address = np.nan
            ISD_status = np.nan
            count_listings = np.nan
            license_cat_max_exceed = np.nan
            if 'STR' in license: # indicates exemption claimed
                ISD_category = license_cat_dict[license]['name'].replace('(verified)', '(not verified)')
                status = simple_status = "Exempt (not verified)"
                full_status = f"Exempt (not verified): {ISD_category.split(': ')[1]}"
            elif 'Missing' in license: # no license claimed
                ISD_category = np.nan
                status = 'No license claimed' 

        # license number CLAIMED in airbnb listing (non-exempt & non-missing)
        else:    
            count_listings = len(listings_license_data)
            
            # claimed license FOUND in ISD DATA: (MATCH)
            if license in isd['license_num'].values:  
                isd_license_data = isd[isd.license_num==license]
                
                # define given license details from ISD match
                found_status = isd_license_data['Status'].values[0]
                address = isd_license_data['Address'].values[0]
                cat = isd_license_data['Category'].values[0]
                category_dict = license_cat_dict[cat]
                ISD_category = category_dict['name']
      
                # CHECK FOR EXCEEDED LIMIT OF LISTINGS OR ACCOMMODATIONS....
                # (based on license type and the number of accoms/listings using this license)
                # From the City Ordinance:
                #   "Occupancy shall be limited to five bedrooms or ten
                #    guests in a Home Share Unit, WICHEVER IS LESS."
                
                # so check if the lesser of accomms or n_listings exceeds the license limit:
                count_accomms = listings_license_data.accommodates.sum()
                license_cat_max_exceed = np.nan
                if count_listings <= count_accomms:
                    # do beds(≈n_listings) exceed category limit?
                    if count_listings > category_dict['maxbeds']:
                        license_cat_max_exceed = 'listing'
                elif count_accomms <= count_listings: 
                    # do guests(≈accomdations) exceed category limit?
                    if count_accomms > category_dict['maxguests']:
                        license_cat_max_exceed = 'guest'
                
                
                # define full details of license status (FULL version and SIMPLE version of STATUS)
                if 'Active' in found_status:   
                    ISD_status = found_status
                    ISD_address = address
                    if 'Exempt' in ISD_category:
                        status = simple_status = "Exempt (verified)"
                        full_status = f"{status}: {ISD_category.split(': ')[1]}"
                    elif str(license_cat_max_exceed)!='nan':
                        status = simple_status = full_status = 'Active (limit exceeded)'
                    else: status = 'Active'
                
                else:
                    status = 'Expired/Void/Revoked' # group these for SIMPLE version
                    ISD_status = full_status = found_status # use exact non-active type for FULL version
                    ISD_address = np.nan
                    
            # NO MATCH FOUND IN ISD FOR THIS CLAIMED LICENSE:
            else: 
                status = 'Not found (fabricated)' 
                ISD_status = status
                ISD_address = np.nan
                ISD_category = np.nan
                license_cat_max_exceed = np.nan
                
    # ----- (still iterating through this license...)
        
        # assign the new custom-defined license details back to all listings that use this license
        listings.loc[listings_license_data.index, 'status'] = status
        listings.loc[listings_license_data.index, 'ISD_status'] = ISD_status
        listings.loc[listings_license_data.index, 'ISD_address'] = ISD_address
        listings.loc[listings_license_data.index, 'ISD_category'] = ISD_category
        listings.loc[listings_license_data.index, 'license_listing_count'] = count_listings
        listings.loc[listings_license_data.index, 'license_cat_max_exceed'] = license_cat_max_exceed

        # assign multiple versions of "status" (SIMPLE and FULL)
        # for displaying levels of detail in different parts of the site
        if simple_status:
            listings.loc[listings_license_data.index, 'simple_status'] = simple_status
        else: simple_status = status
        if full_status:
            listings.loc[listings_license_data.index, 'full_status'] = full_status
        listings.loc[listings_license_data.index, 'simple_status'] = simple_status

    #——————— END LOOP OVER THIS LICENSE
    
    # use SIMPLE status for licenses where no FULL status was found
    listings.full_status = listings.full_status.fillna(listings.simple_status)
    print(f"    > Matched {listings.ISD_address.notna().sum()} to ISD data")

    # log percentage of active-licensed listings:
    pct_active = round(listings.status.value_counts(normalize=True)['Active']*100)
    print(f"  >>> {int(pct_active)}% of total listings have an Active license")
    print(' ', '—'*30)
    
    # check if license numbers are used by multiple hosts
    # this function has unecessary redundancy... but still finishes quickly enough 
    print("  > Searching for license details of listings in ISD data...")  
    def find_shared_licenses(listings):
        for host in tqdm(set(listings.host_id)): # isolate data for each host 
            host_data = listings[listings.host_id==host]
            # iterating host-license groups allows exclusion of the self-host from list of others
            for license in host_data.license.unique(): # isolate data for each license by host
                group_idx = host_data[host_data.license==license].index
                if 'STR' not in license and 'Missing' not in license:
                    all_hosts = set(listings[listings.license==license].host_id.unique())
                    if len(all_hosts)>1: # more than one host using this license
                        all_hosts.remove(host) # [should assign to all hosts using this license...]
                        listings.loc[group_idx, 'other_hosts'] = ', '.join([str(h) for h in all_hosts])     
                else: listings.loc[group_idx, 'other_hosts'] = np.nan
        return listings
    print('  > Defining licenses used by multiple hosts ')
    listings = find_shared_licenses(listings)
    
    
    # define a common index for the airbnb data and the ISD data (for future reference back to ISD matches)
    # not currently needed / used, but may be useful in the future
    def get_isd_index(listings):
        print('Mapping exact index back to listings dataframe...') 
        isd_match_idx = {}
        for license in listings.license.unique(): # save time   
            list_idx = listings[listings.license==license].index
            if 'Exempt' in license or 'Missing' in license: continue
            matches = ''
            for isd_idx in isd.index:
                if license in isd.loc[isd_idx, 'license_num']:
                    matches+=str(isd_idx)+' '

            listings.loc[list_idx, 'isd_index'] = matches

    # will be typing this a lot.
    listings = listings.rename(columns={'host_name':'host'})
    print('—'*60)
    
    # drop lots of unecessary information
    keep_cols = ['id', 'listing_url', 'name', 'host_id', 'host_url',
       'host', 'neighbourhood', 'neighbourhood_cleansed', 'last_scraped',
       'neighbourhood_group_cleansed', 'latitude', 'longitude',
       'property_type', 'room_type', 'accommodates', 'bedrooms',
       'beds', 'amenities', 'minimum_nights', 'maximum_nights',
       'minimum_minimum_nights', 'has_availability', 'price',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'number_of_reviews', 'license', 'instant_bookable',
       'host_alias', 'alias_id', 'status', 'ISD_status', 'ISD_address',
       'ISD_category', 'license_listing_count', 'license_cat_max_exceed', 'simple_status',
       'full_status', 'other_hosts']
    for c in listings.columns:
        if c not in keep_cols:
            listings.drop(columns=[c], inplace=True)
    
    
    # DONE LOADING & PARSING LISTINGS.
    return listings




# # # # # # # #  CLUSTER  # # # # # # # #

# In THREE main phases (all the code for the rest of this file) the listings are clustered:
#   1. Start by grouping listings by host, then grouping the listings for each host by the claimed license.
#        - Many hosts use one license across all listings in one given building
#        - But sometimes, a license is used in multiple buildings. So we need search for sub-clusters within these groups
#      ——> FIND CLUSTERS for each HOST-LICENSE group
#   2.  Identify overlapping clusters from phase one
#        - Two high-density clusters, each with many listings, probably represent two distinct buildings.
#        - On the other hand, 4 or 5 low-density, low-volume clusters by the same host may represent one very anonymized building.
#      ——> FIND PARENT CLUSTERS for host ALL SUB-CLUSTERS for the host
#   3.  Identify POSSIBLE sub-clusters
#        - In RARE cases, a high-volume parent cluster may still actually represent multiple buildings. AKA the initital two phases failed
#        - So we check for strong indications of sub-clustering. If any high-confidence sub-clusters are present,
#                                                                save them, but don't alter the original centroid.
#      ——> CHECK FOR POSSIBLE SUB-CLUSTERS for each CLUSTER group

# After the first two phases of clustering, we will have CENTROIDS - a dataframe of parent clusters and their radial statistics
# After the third phase of clustering, we willl get SUB_CENTROIDS - a dataframe of possible sub-clusters and their radial statistics
# ... plus the final LISTINGS dataframe, with index numbers for each listing indicating the centroid (and sub_centroid when present)

# convert a distance from FT to (approximately) lat/long — for setting a dynamic epsilon value in DBSCAN 
# messy but necessary because DBSCAN requires the same unit input as it scans over (coordinates in this case)
    # (epsilon will shift as the target location shifts laterally. but only barely!)
def eps_calc(desired_ft):
    # these values custom-defined for 500ft using Google Maps measuring tape tool in the Boston public garden

    cal = [ (42.354304, -71.069223),  # 500 ft calibration
            (42.353843, -71.070958) ]
    # using calibration, calculate foot-factor
    x_diff = (cal[0][0] - cal[1][0])**2
    y_diff = (cal[0][1] - cal[1][1])**2
    ft_factor = np.sqrt(x_diff + y_diff) / 500 # calibration
    epsilon = round(desired_ft*ft_factor, 8)  # convert using calculated foot-conversion factor
    return epsilon


# explode the initial listings into sub-clusters, separated by host and claimed license
# AKA find listing sub-clusters within host-license groups, or multiple buildings using the same license

def create_sub_layer(listings, epsilon, min_samples, feature, sub_feature, epsilon2=False, min_samples2=False, verb=False):
    
    # log
    print('CREATING CLUSTER LAYER under', feature, '...')
    print('—'*60)
    classes_created = 0
    recognized_centroids = 0
    new_centroids = 0
    
    # USE CACHE TO MAINTAIN THE SAME HOST NUMBER ACROSS MUTLIPLE SCRAPES (over time)
    # get static host name index for top layer (so hosts always have the same apparent #)
    fname = f'{feature}_index.cache'
    if feature=='host_id' and fname in os.listdir('cache'):
        with open(f'cache/{fname}','rb') as file:
            feat_index = pickle.load(file)
            static_indexer = max(v['static_index']
                                 for v in feat_index.values())
    else:
        static_indexer = 1
        feat_index = {}
    print(f'  > {len(feat_index)} {feature} index names found in cache')
    

    # start iterating over top-layer groups
    print(f'  > DBSCANning with {min_samples} min_samples | epsilon = {epsilon} ft.')
    found = 0
    new_hosts = 0
    
    key_order = listings[feature].value_counts().index # start with the biggest groups. 
    for unique_parent in tqdm(key_order):
        
        # don't sub-scan outlier clusters
        if 'sub' in sub_feature:
            if '-1' in str(unique_parent): 
                continue
            static_index = unique_parent
            
        # get static host ID for top layer
        else:       # take host id from cache
            if str(unique_parent) in feat_index.keys():
                static_index = feat_index[str(unique_parent)]['static_index']
                found += 1
            else:   # create new host id on top of cache
                new_hosts += 1
                static_index = static_indexer
                feat_index[str(unique_parent)] = { # save to cache
                    'static_index': static_indexer }                
                static_indexer += 1
                        
        # define data of target group
        parent_group = listings[listings[feature]==unique_parent]
        # for main host-license groups
        if 'host' in feature:
            c_counter = 0 # for class counter
            for license in parent_group.license.unique():
                parent_license_data = parent_group[parent_group.license==license]
                classes = dbscan(
                        parent_license_data[['latitude', 'longitude']].values,
                        eps = eps_calc(epsilon), # convert feet to meters (unit used in folium)
                        min_samples = min_samples, # min 3 together only
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=30,
                        p=2,)[1]
                                
            # name found centroids using static index technique:
                # reindex found classes for all license groups in this host
                new_classes = classes.copy()
                dbscan_found = list(set(classes))
                
                if -1 in dbscan_found:  # leave outliers as -1
                    dbscan_found.remove(-1)
                    
                already_found_centers = {}
                for c in dbscan_found: # for each class, iterate host centroids counter
                    c_idx = [ i for i, v in enumerate(classes) if v == c ]
                    c_counter += 1
                    for res_row in c_idx:
                        new_classes[res_row] = c_counter # count found centroids for host
                                            
                dbscan_found = list(set(new_classes))
                # assign top-layer find
                listings.loc[parent_license_data.index, sub_feature+'_found'] = new_classes
                            
              
        # [all licenses done]
            listings.loc[parent_group.index, sub_feature # default fill with found
                        ] = listings.loc[parent_group.index, sub_feature+'_found']
            
            
            # check for distances between classes (group multi-license centroids) to join overlapping groups
            parent_group = listings[listings[feature]==unique_parent] # re-define
            found_groups = parent_group[sub_feature+'_found'].unique()
            found_host_centers = {}
            centroid_num = 1
            for found_group in found_groups:
                if '-1' in str(found_group): continue # outliers
                    
                fg_data = parent_group[parent_group[sub_feature+'_found']==found_group] # fg data
                found_center = (fg_data.latitude.mean(), fg_data.longitude.mean())
                
                # make bounding box for existing centroid
                fg_cc_tl = (fg_data.latitude.max(),  fg_data.longitude.max())
                fg_cc_br = (fg_data.latitude.min(),  fg_data.longitude.min())
                fg_max_range = geodesic(fg_cc_tl, fg_cc_br).feet*.5
                
                # check if centroid overlaps with other centroids
                overlap_class = False
                for af_class, af_center in found_host_centers.items():
                    distance = geodesic(found_center, af_center).feet
                    
                    # this centroid is within the bounding box of an already-found centroid
                    if distance < fg_max_range: 
                        this_class = overlap_class = af_class # so just join them 
                        break
                
                # unique new group
                if not overlap_class:
                    this_class = centroid_num
                    found_host_centers[this_class] = found_center
                    centroid_num += 1

                    
                 # assign, host-centroid was found 
                listings.loc[fg_data.index, sub_feature] = f"{static_index}_{int(this_class)}"
            
            listings.drop(columns=[sub_feature+'_found'], inplace=True) # dont need this anymore
            
            
        # CHECK FOR ALREADY-RECOGNIZED CENTROIDS to establish centroid names...
            # search for new centroids, and use the EXISTING NAME from centroids in the PREVIOUS SCRAPE
            # IF a centroid is within a minimum distance and within a minimum size ratio of the old one
            # ELSE (no matching centroid found or too much change from old scrape) --> create new centroid name

            if 'centroids' not in list(feat_index[str(unique_parent)].keys()):
                feat_index[str(unique_parent)]['centroids'] = {} 
                
            # list available centroid numbers for this host (not already used by pre-centroid)
            centroids_avail = list(range(1, 100))
            for p_cen in list(feat_index[str(unique_parent)]['centroids'].keys()):
                centroids_avail.remove(int(p_cen)) # labels already taken in from scrape
            
            # re-define group data to get existing centroid names found
            parent_group = listings[listings[feature]==unique_parent]
            existing_groups = parent_group[sub_feature].value_counts().index # centroids sorted by size   
            non_outlier_groups = [g for g in existing_groups if '-1' not in str(g)]
            if verb and len(non_outlier_groups):
                print('\n', '—'*60)
                print(unique_parent)
            
            # these statistical variables will be used to determine whether a centroid is new or recognized
            n_listings_change = {} # to store amount change in listings for each centroid
            feat_index_temp = {} # for temporarily storing data on new centroids
            
            for e_group in existing_groups: # ITERATE NEWLY FOUND CENTROIDS GROUPS
                if '-1' in str(e_group): continue

                e_data = parent_group[parent_group[sub_feature]==e_group]
                e_center = (e_data.latitude.mean(), e_data.longitude.mean())
                e_cc_tl = (fg_data.latitude.max(),  fg_data.longitude.max())
                e_cc_br = (fg_data.latitude.min(),  fg_data.longitude.min())
                e_max_range = geodesic(e_cc_tl, e_cc_br).feet*.5 # max search distance for existing centroids
                e_n_listings = len(e_data)
                
                if verb: print(f'\n> {e_group} ({e_n_listings}):  {e_center}')
                
                 # check all OLD host centroids for matches to this NEW centroid...
                recognized = False
                for pre_centroid in feat_index[str(unique_parent)]['centroids'].keys():
                    
                    # same as above, define statistics for existing centroid against pre-cached centroid
                    p_center = feat_index[str(unique_parent)]['centroids'][pre_centroid]['center']
                    p_n_listings = feat_index[str(unique_parent)]['centroids'][pre_centroid]['n_listings']
                    p_max_range = feat_index[str(unique_parent)]['centroids'][pre_centroid]['max_range']
                    dist_diff = geodesic(e_center, p_center).feet        
                    n_listings_diff = p_n_listings-e_n_listings
                    n_listings_pct_diff = (n_listings_diff/e_n_listings)
                    
                    feat_index_temp[pre_centroid] = { # store newfound centroid data stats
                        'n_listings': p_n_listings,
                        'center': p_center,
                        'max_range':p_max_range,
                    }
                    
                    # overlapping bounding box locations
                    if dist_diff <= max(e_max_range, p_max_range): # check the bigger of both bounding boxs
                        
                        # w.in 75% difference in listing count
                        if abs(n_listings_pct_diff) < .75:
                            
                            recognized = True
                            recognized_centroids += 1
                            this_class = pre_centroid  # assign existing number for centroid
                            listings.loc[e_data.index, sub_feature+'_change_n_listings'] = n_listings_diff
                            listings.loc[e_data.index, sub_feature+'_change_pct_listings'] = n_listings_pct_diff
                            if verb:
                                print('  > RECOGNIZED', e_group, '——>', pre_centroid)
                                print(f'    > {pre_centroid} ({p_n_listings}):  {p_center}')
                                print('    > DISTANCE:', dist_diff)
                            break # stop searching, assumed found (possible errors here...)**

                if not recognized: # no matched found from pre-existings centroids
                    new_centroids += 1
                    this_class = str(centroids_avail[0]) # assign new number for centroid
                    centroids_avail = centroids_avail[1:]
                    if verb:
                        print('  > NOT RECOGNIZED', e_group, '——>', this_class)
                        
                # save new matching centroid number to temporary cache (over old centroid number)
                feat_index_temp[this_class] = {
                    'n_listings': e_n_listings,
                    'center': e_center,
                    'max_range': e_max_range, }
                if verb: print(f'    > {static_index}_{int(this_class)} data saved to static index')
                    
                # assign final centroid labels for this host:
                listings.loc[e_data.index, sub_feature] = f"{static_index}_{int(this_class)}"
                # bool for recognized vs. new centroid:
                listings.loc[e_data.index, sub_feature+'_recognized'] = recognized
            
            # ———————————————
            # finished with ALL HOSTS hosts  (all clustering done for top-level centroids)
            # overwrite permanent cache with new data for recognized centroids
            # for old centroids not seen, keep old data & label in case of re-appearance.
            
            if not config.just_host_index:
                feat_index[str(unique_parent)]['centroids'] = feat_index_temp
                
            
        # ——————————————
        # for sub-centroids:    
        else: # try granular then broad search, save best results (most classes found)
            classes = dbscan(
                        parent_group[['latitude', 'longitude']].values,
                        eps = eps_calc(epsilon), # convert feet to meters (unit used in folium)
                        min_samples = min_samples, # min 3 together only
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=30,
                        p=2,)[1]

            classes2 = dbscan(
                    parent_group[['latitude', 'longitude']].values,
                    eps = eps_calc(epsilon2), # convert feet to meters (unit used in folium)
                    min_samples = min_samples2, # min 3 together only
                    metric='minkowski',
                    algorithm='auto',
                    leaf_size=30,
                    p=2,)[1]
            if len(set(classes2))>=len(set(classes)): # more or larger same num clusters found from broader searchs
                classes = classes2
                
                
            # assign sub-find
            listings.loc[parent_group.index, sub_feature
                        ] = [f"{static_index}_{c}" for c in classes]
   
    if 'sub' not in sub_feature:
        print(f'Hosts: found {found} in cache | {new_hosts} new')
        
    not_clustered = listings[sub_feature].str.contains('-1').sum()
    print('  >', not_clustered, 'outliers not clustered')
    created = listings[sub_feature].nunique()
    print('  >', created, 'sub-classes generated & re-assigned')
    print(' ', '—'*30)
    print('  >', new_centroids, 'new centroids identified')
    print('  >', recognized_centroids, 'centroids recognized from cache')



    if 'sub' not in sub_feature:
        # save new host id to cache
        with open(f'cache/{fname}','wb') as file:
            pickle.dump(feat_index, file)
    print(' ', '—'*30)
    return listings



# assign colors & stats to groups (for mapping)
def get_layer_clusters(listings, feature):

    clusters = {}
    print('  > Getting cluster params...')
    for cluster in tqdm(listings[feature].dropna().unique()):
        
        if '-1' in str(cluster): # ignore outliers
            continue
        
        # get cluster stats
        c_data = listings[listings[feature]==cluster]
        
        max_lat = c_data.latitude.max()
        max_lng = c_data.longitude.max()
        min_lat = c_data.latitude.min()
        min_lng = c_data.longitude.min()
        mean_lat = c_data.latitude.mean()
        mean_lng = c_data.longitude.mean()

        radius = geodesic((max_lat,max_lng), (min_lat,min_lng)).meters
        radius = max(radius, 80)

        clusters[cluster] = {
            'radius': radius*.65,
            'mean_latitude': mean_lat,
            'mean_longitude': mean_lng,}

        # name + rectangle bounding box 
    print(f'  > Found {len(clusters)} {feature}s')
    return clusters



# generate a df with subgroup centroids as rows and group info + geo stats as columns
def merge_clusters(centroids, listings, feature):
    print(' ', '—'*30)
    
    print(f'  > Calculating {feature} stats for {len(centroids)} clusters')
    # compile dicts of info about each centroid
    centroid_data = []
    
    for c_group in tqdm(centroids.keys()): # iterate through centroid groups
        if '-1' in str(c_group):
            continue
            
        c_data = centroids[c_group]
        l_data = listings[listings[feature]==c_group]
        
        # basic info about centroid group
        if l_data.license_cat_max_exceed.notna().sum()>0:
            license_exceed_type = l_data.license_cat_max_exceed.value_counts().index[0]
        else: license_exceed_type = np.nan
            
        def clean_dict(in_d):
            def clean(string):
                return str(string).replace('.0', '').replace("'", '"').replace(':', '-')
            return {clean(k): clean(v) for k,v in in_d.items()}

        centroid = {feature: c_group,
                    'count_listings': len(l_data),
                    'count_licenses': l_data.license.nunique(),
                    'license_dict': clean_dict(dict(l_data.license.value_counts())),
                    'status_dict': clean_dict(dict(l_data.status.value_counts())),
                    'license_exceed_type': license_exceed_type,
                    'minimum_nights': l_data.minimum_minimum_nights.min(),
                    'ISD_address_dict':clean_dict(dict(l_data.ISD_address.value_counts())),
                    'host': l_data.host.values[0],
                    'host_id': l_data.host_id.values[0],
                    'alias_dict': clean_dict(dict(l_data.alias_id.value_counts())),
                    'shared_license': l_data.other_hosts.notna().sum() > 0,
                    'change_n_listings': l_data.centroid_group_change_n_listings.values[0]
                   }
        if not config.old:
            centroid['avg_USD/night'] = round(l_data['price'].mean(), 2)
        if 'sub' in feature: centroid['centroid_group'] = l_data.centroid_group.values[0]
        # add pre-calc'ed centroid stats
        for k, v in c_data.items(): centroid[k] = v
        
        centroid_data.append(centroid)
    return pd.DataFrame(centroid_data).sort_values('count_listings', ascending=False)


# calc detailed radial statistics for each centroid based on geo listing dispersal
def merged_radial_stats(centroids, listings):
    print(' ', '—'*30)
    feature = centroids.columns[0]
    print(f'  > Analyzing {feature} radial stats...')
    
    distances = []
    print('  > Calculating listing distances from associated centroids...')
    for l_idx in tqdm(listings.index): # iterate listings
        c_group = listings.loc[l_idx, feature]
        
        if '-1' in str(c_group) or str(c_group)=='nan': continue # ignore outliers
            
        else:
            l_lat = listings.loc[l_idx, 'latitude']
            l_lng = listings.loc[l_idx, 'longitude']
            c_listings = listings[listings[feature]==c_group]
            c_idx = centroids[centroids[feature]==c_group].index
            c_lat = centroids.loc[c_idx, 'mean_latitude'].values[0]
            c_lng = centroids.loc[c_idx, 'mean_longitude'].values[0]
            distance = geodesic((c_lat,c_lng), (l_lat,l_lng)).feet
            listings.loc[l_idx, f'{feature}_distance'] = distance
                
    # calculate location confidence (using avg. distance from centroid)
    print(' ', '—'*30)
    print('  > Calculating centroid location confidence...')
    distance_dict = {}
    distance_radius = {}
    for c_group in tqdm(centroids[feature].unique()):  # iterate centroids
        
        listings_data = listings[listings[feature]==c_group] 
        c_idx = centroids[centroids[feature]==c_group].index
        
        max_distance = round(listings_data[f'{feature}_distance'].max(), 5)
        if len(listings_data)<3:  
            centroids.loc[c_idx, 'confidence'] = 'unknown (too few)'
        elif max_distance < 50:
            centroids.loc[c_idx, 'confidence'] = 'unknown (too close)'
        else:
            centroids.loc[c_idx, 'confidence'] = f'within {round((max_distance)/50)*50} ft.'
        
        centroids.loc[c_idx, 'max_listing_distance'] = max_distance
        
        centroids.loc[c_idx, 'avg_listing_distance'] = round(
            listings_data[f'{feature}_distance'].mean(), 4)
        centroids.loc[c_idx, 'std_listing_distance'] = round(
            listings_data[f'{feature}_distance'].std(), 4)

    return centroids, listings


# get "best-guess" addresses from lat long 
def geocode(centroids):
    print('Reverse geocoding cluster centroids with nominatim')
    print(' ', '—'*30)
    centroids = centroids.copy()
    feature = centroids.columns[0]
    
    fname = 'geocode.cache'
    if fname in os.listdir('cache'):
        with open(f'cache/{fname}', 'rb') as file:
            geo_dict = pickle.load(file)
    else: geo_dict = {}
    print(f'  > Found {len(geo_dict)} pre-geocoded locations in cache')
    
    cache_found = 0
    new_find = 0
    for ridx in tqdm(centroids.index):
        
        if '-1' in centroids.loc[ridx, feature]: continue # don't scan outliers
            
        key = f"{centroids.loc[ridx,'mean_latitude']}, {centroids.loc[ridx,'mean_longitude']}"
            
        if key in geo_dict.keys(): cache_found += 1            
        else:
            new_find += 1
            try:
                georeq = geolocator.reverse(key, addressdetails=True)
                address = georeq.raw['address']
            except:
                with open(f'cache/{fname}','wb') as file:
                    pickle.dump(geo_dict, file)
                raise RuntimeError('Connection lost — cache autosaved!')
                
            geo_dict[key] = address
            time.sleep(.1)
        
        for component in geo_dict[key].keys():
            centroids.loc[ridx, f'GCODE_{component}'] = geo_dict[key][component]

    # log & save to cache
    print(f'  > {cache_found} retrieved from the cache')
    print(f'  > {new_find} new locations geocoded & added to cache')
    print(f'  > Geocode completed')
    print('—'*60)

    with open(f'cache/{fname}','wb') as file:
        pickle.dump(geo_dict, file)
    
    return centroids

# This function is for internal testing only:
# FOR VISUAL ANALYSIS OF CLUSTERING:
def map_corp_group(n, listings, centroids, sub_centroids):
        # THIS DOES NOT MAP NON-CLUSTERED POINTS

    sc_count = 0
    host_id = f'CorpGroup{n}'
    host_data = listings[listings.host_id==host_id]
    

    start_coords = [42.343, -71.085]
    m = folium.Map(location=start_coords, tiles=None, zoom_start=13)
    folium.TileLayer('cartodbpositron', control=False).add_to(m)


    main_cluster_layer = folium.FeatureGroup(
            name=f'main clusters', control=True).add_to(m)
    sub_cluster_layer = folium.FeatureGroup(
            name=f'sub clusters', control=True).add_to(m)

    feature = 'centroid_group'
    sub_feature = 'sub_centroid'
    # TOP CLUSTERS
    for top_cluster in host_data[feature].unique():
        
        if str(top_cluster)=='nan': continue # don't map outlier cluster 
        c_data = listings[listings[feature]==top_cluster]
        c_index = centroids[centroids[feature]==top_cluster].index
        c_stats = centroids.loc[c_index]
        

        for l_idx in c_data.index:
            l_row = c_data.loc[l_idx]
            folium.vector_layers.Circle(
                location = [l_row.latitude, l_row.longitude],
                color = 'black' if str(top_cluster)!='nan' else 'red',
                radius = '10',
                tooltip = folium.Tooltip(top_cluster)
            ).add_to(main_cluster_layer)


        folium.vector_layers.Circle(
            location = [c_stats['mean_latitude'].values[0],
                        c_stats['mean_longitude'].values[0]],
            color = 'cadetblue',
            fill = True,
            radius = c_stats['radius'].values[0],
            tooltip = folium.Tooltip(top_cluster)
        ).add_to(main_cluster_layer)


    # SUB CLUSTERS  
        sub_clusters = [c for c in c_data[sub_feature].unique() if str(c)!='nan']
        if len(sub_clusters)>1:
            for sub_cluster in sub_clusters:
                sc_count += 1
                sub_c_index = sub_centroids[sub_centroids[sub_feature]==sub_cluster].index
                sub_c_stats = sub_centroids.loc[sub_c_index]
                folium.vector_layers.Circle(
                        location = [sub_c_stats['mean_latitude'].values[0],
                                    sub_c_stats['mean_longitude'].values[0]],
                        color = 'blue',
                        fill = True,
                        radius = sub_c_stats['radius'].values[0],
                        tooltip = folium.Tooltip(sub_cluster)
                    ).add_to(sub_cluster_layer)

    print(host_data[feature].nunique(), 'clusters |', sc_count, 'sub-clusters')


    folium.LayerControl(collapsed=False).add_to(m)
    return m, sc_count


# Run all functions in this notebook. Variables allow specifying which data to focus on (current vs. old scrape)
def run(verb=False, sample_test=False, old=False, just_host_index=False):
    if old: config.old=True
    else: config.old=False
        
    if just_host_index: config.just_host_index=True
    else: config.just_host_index=False
        
    listings = load_parsed_listings(sample_test=sample_test)

    # main clusters
    eps = 300
    min_samples = 2
    feature = 'host_id'
    sub_feature = 'centroid_group'
    listings = create_sub_layer(listings, eps, min_samples, feature, sub_feature, verb=verb)
    if config.old:
        print('Old centroids identified.\n\n')
        return
    if config.just_host_index:
        print('Host index established (centroids not saved).\n\n')
        return
    centroids = get_layer_clusters(listings, sub_feature)
    centroids = merge_clusters(centroids, listings, 'centroid_group')
    centroids, listings = merged_radial_stats(centroids, listings)
    centroids = geocode(centroids)

    # sub clusters 
    eps, eps2 = 70, 135
    min_samples, min_samples2 = 5, 4
    feature = 'centroid_group' # building smaller from main centroids
    sub_feature = 'sub_centroid' 
    listings = create_sub_layer(listings, eps, min_samples, feature, sub_feature,
                                         eps2, min_samples2) 
    sub_centroids = get_layer_clusters(listings, sub_feature)
    sub_centroids = merge_clusters(sub_centroids, listings, 'sub_centroid')
    sub_centroids, listings = merged_radial_stats(sub_centroids, listings)
    sub_centroids = geocode(sub_centroids)
    na_outliers = lambda x: x if '-1' not in str(x) else np.nan
    listings[feature] = listings[feature].apply(na_outliers)
    listings[sub_feature] = listings[sub_feature].apply(na_outliers)
    
    listings.drop(columns=['centroid_group_change_n_listings'], inplace=True)
    
    listings.to_csv('clustered/listings.csv', index=False)
    centroids.to_csv('clustered/centroids.csv', index=False)
    sub_centroids.to_csv('clustered/sub_centroids.csv', index=False)

    # verify / complete
    print('—'*30, '\nClustering completed\n'+('—'*30))


