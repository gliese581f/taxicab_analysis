def create_polygon(nyc_shapefile,region):
    
    from shapely.geometry import Polygon
    import fiona 
    import pyproj
    import shapely.ops as ops
    from functools import partial
    from shapely.ops import cascaded_union
    from shapely.prepared import prep
    
    place    = []
    boroughs = ['Queens','Manhattan','Brooklyn','Bronx','Staten Island']
 
    shapefile_obj = fiona.open(nyc_shapefile)

    for regions in range(len(shapefile_obj)): 
        if shapefile_obj[regions]['properties']['neighborho'] == str(region):
            place = Polygon(shapefile_obj[regions]['geometry']['coordinates'][0])
        elif region in boroughs:
            if shapefile_obj[regions]['properties']['borough'] == str(region):
                place.append( Polygon(shapefile_obj[regions]['geometry']['coordinates'][0]) )
            
    poly = cascaded_union(place)
    
    poly_transform = ops.transform(partial(pyproj.transform,pyproj.Proj(init='EPSG:4326'),pyproj.Proj(proj='aea',lat1=poly.bounds[1],lat2=poly.bounds[3])),poly)
    
    return prep(poly), poly_transform.area


def region_to_region_data(dataframe,nyc_shapefile,region1,region2):
    
    from shapely.geometry import Point, Polygon
    import pandas as pd
    
    if region1 != region2:
        poly1 = create_polygon(nyc_shapefile,region1)[0]
        poly2 = create_polygon(nyc_shapefile,region2)[0]
    else:
        poly1 = create_polygon(nyc_shapefile,region1)[0]
        poly2 = poly1
       
    dataframe_filter = dataframe.apply( lambda row: poly1.contains( Point(row['pickup_longitude'],row['pickup_latitude'])) & poly2.contains(Point(row['dropoff_longitude'],row['dropoff_latitude'])),axis=1 )

    return dataframe.drop(dataframe[~dataframe_filter].index)


def velocity(trip_subset,min_triptime,min_tripdistance):
    
    import numpy as np
    import datetime
    
    stat =  np.array(trip_subset[(trip_subset['trip_time_in_secs'] >min_triptime) & (trip_subset['trip_distance'] >min_tripdistance)].sort_values('pickup_datetime')['trip_distance']/(trip_subset[(trip_subset['trip_time_in_secs'] >min_triptime)& (trip_subset['trip_distance'] >min_tripdistance)].sort_values('pickup_datetime')['trip_time_in_secs']))
    
    dates = np.array(trip_subset[(trip_subset['trip_time_in_secs'] >min_triptime)& (trip_subset['trip_distance'] >min_tripdistance)].sort_values('pickup_datetime')['pickup_datetime'])

    days  = [ datetime.datetime(int(dates[ii][0:4]),int(dates[ii][5:7]),int(dates[ii][8:10]),int(dates[ii][11:13])).day for ii in range(len(dates))]
    hours =  [ datetime.datetime(int(dates[ii][0:4]),int(dates[ii][5:7]),int(dates[ii][8:10]),int(dates[ii][11:13])).hour for ii in range(len(dates))]
    
    return stat,dates,days,hours


def mean_density(stat,days,hours,obj1,obj2):
    
    import numpy as np

    rides = [ stat[entry] for entry in range(len(stat)) if days[entry] in obj1 and hours[entry] in obj2 ]
    
    number = len(rides)
    mean = np.mean(rides)
   
    return mean, number   


def aggregate(stat,days,hours,obj1,obj2, time_obj):
    
    import numpy as np
    
    total, total_density = [], []
    
    if time_obj   == 'hours':
        for times in np.arange(1,24,1):
            total.append( mean_density(stat,days,hours,obj1,[times])[0] )
            total_density.append( mean_density(stat,days,hours,obj1,[times])[1] )
        
    elif time_obj == 'days':
        for times in np.arange(1,30,1):
            total.append( mean_density(stat,days,hours,[times],obj2)[0] )
            total_density.append( mean_density(stat,days,hours,[times],obj2)[1] )

    return np.array(total),np.array(total_density)


def radial_distribution(trip_subset,pickup,dropoff,sample):
    
    import shapely.geometry
    import numpy as np
    from shapely.geometry import Point
    from itertools import combinations
    import random
    
    pairwise_distances = []
    random_indices = [ int( len(trip_subset)*random.random() ) for _ in range(sample) ]
    pairs = list(combinations(random_indices,2))
   
    for p1 in pairs:
        point1 = Point(np.array(trip_subset[str(pickup)+'_longitude'])[p1[0]],np.array(trip_subset[str(pickup)+'_latitude'])[p1[0]])
        point2 = Point(np.array(trip_subset[str(dropoff)+'_longitude'])[p1[1]],np.array(trip_subset[str(dropoff)+'_latitude'])[p1[1]])
        pairwise_distances.append( point1.distance(point2) )
    
    return pairwise_distances 




def joint_distribution(trip_subset,min_dist,sample,obj1,obj2):
    
    pickup = 'pickup'
    dropoff = 'dropoff'
    
    import shapely.geometry
    import numpy as np
    from shapely.geometry import Point
    from itertools import combinations
    import random
    
    shared_rides = []
    
    times = [ efficiency(trip_subset, 120,0.25)[2], efficiency(trip_subset,120,0.25)[3] ]
    time_sorted = [ entry for entry in range(len(times[0])) if times[0][entry] in obj1 and times[1][entry] in obj2 ]
    
    if time_sorted > sample: time_sorted = time_sorted[:sample]
    
    pairs = list(combinations(time_sorted,2))
    
   
    for p1 in pairs:
        point1 = Point(np.array(trip_subset[str(pickup)+'_longitude'])[p1[0]],np.array(trip_subset[str(pickup)+'_latitude'])[p1[0]])
        point2 = Point(np.array(trip_subset[str(pickup)+'_longitude'])[p1[1]],np.array(trip_subset[str(pickup)+'_latitude'])[p1[1]])
        point3 = Point(np.array(trip_subset[str(dropoff)+'_longitude'])[p1[0]],np.array(trip_subset[str(dropoff)+'_latitude'])[p1[0]])
        point4 = Point(np.array(trip_subset[str(dropoff)+'_longitude'])[p1[1]],np.array(trip_subset[str(dropoff)+'_latitude'])[p1[1]])
        if point1.distance(point2) <= min_dist and point3.distance(point4) <= min_dist:
            shared_rides.append( [point1,point2,point3,point4] )
    
    return shared_rides

def agg_joint(trip_subset,dist,day):
    joints = []

    for hour in np.arange(0,24,1):
        joints.append( len(joint_distribution(trip_subset,dist,100,[day],[hour]) ))
           
    return joints
