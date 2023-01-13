import numpy as np
from skimage.measure import label, regionprops
from math import sqrt, pi


class MaskInspection:
    def __init__(self,
                 mask,
                 tr: float):
        
        self.mask = mask
        self.tr = tr    

    def coordinates_calculation(self):
        ''' id : identifier
            label: label of objects within mask
            x : x coordinate
            y : y coordinate
            r : max radius '''

        info = {'id' :[], 'label':[], 'x':[], 'y':[], 'bbox':[], 'area': [], 'radius':[]}
        id_ = 0

        for L in range(np.min(self.mask),np.max(self.mask)+1):
            
            mask_=np.zeros(self.mask.shape)
            mask_[np.where(self.mask==L)]=1
            lab=label(mask_,background=0)
            props = regionprops(lab)

            for r in props:
                radius=np.sqrt(np.max(np.sum((r.coords-r.centroid)**2,axis=1)))

                if radius>self.tr:
                    m=np.zeros((self.mask.shape[0],self.mask.shape[1]))
                    for c in r.coords:
                        m[c[0],c[1]]=1

                    info['id'].append(id_)
                    info['label'].append(L)
                    info['x'].append(r.centroid[1])
                    info['y'].append(r.centroid[0])
                    info['bbox'].append(r.bbox)
                    info['area'].append(r.area)
                    info['radius'].append(radius)

                    id_+=1
        return info 
            
    def get_dictionaries(self,
                         info:dict):
        '''Get dictionaries of adjacent regions and of regions, setup for Prolog'''
        regions_dict = {}
        adjacent_dict = {}

        for i in range(len(info['id'])):
            regions_dict[info['id'][i]] = {
                'id': info['id'][i],
                'label': info['label'][i],
                'x': info['x'][i],
                'y': info['y'][i],
                'bbox': info['bbox'][i],
                'area': info['area'][i],
                'radius': info['radius'][i],
            }

        regions_dict = self.drop_puddle(regions_dict)    
        # Find the adjacent regions for each region
        for region in regions_dict.values():
            for other in regions_dict.values():
                if region['id'] != other['id'] and region['label'] != other['label']:
                    # Check if the bounding boxes of the regions overlap
                    if region['bbox'][0] <= other['bbox'][2] and region['bbox'][2] >= other['bbox'][0] and region['bbox'][1] <= other['bbox'][3] and region['bbox'][3] >= other['bbox'][1]:
                        # Calculate the distance between the centroids
                        dx = region['x'] - other['x']
                        dy = region['y'] - other['y']
                        d = sqrt(dx**2 + dy**2)
                        # Check if the regions are touching
                        if d < region['radius'] + other['radius']:
                       # Add the region to the list of adjacent regions if it is not already present
                            if region['id'] not in adjacent_dict:
                                adjacent_dict[region['id']] = []
                            if other['id'] not in adjacent_dict[region['id']]:
                                adjacent_dict[region['id']].append(other['id'])

        key_to_rm = ['radius', 'bbox', 'area']
        for region in regions_dict.values():
            for key in key_to_rm:
                try:
                    del region[key]
                except KeyError:
                    pass


        return regions_dict, adjacent_dict



    def drop_puddle(self,
                    regions_dict:dict):
        sum_ = 0
        id_ = []
        for region, item in regions_dict.items():
            if item['label'] == 3:
                sum_ += item['area']  
                id_.append(item['id'])

        if sum_ < (self.mask.shape[0]*self.mask.shape[1]*1e-2):
            for region, item in regions_dict.items():
                if region in id_:
                    item['label']=7

        return regions_dict
    

class savePL:
    def __init__(self,
              regions_dict: dict,
              adjacent_dict: dict):
        self.regions = regions_dict
        self.adjacent = adjacent_dict
        
    def region_pl(self):
        with open('./prolog/facts/regions.pl', 'w') as f:
            for region in self.regions.values():
                f.write(f"region({region['id']}, {region['label']}, {region['x']}, {region['y']}).\n")
            
    def adjacent_pl(self):
        with open('prolog/facts/adjacent.pl', 'w') as f:
            for region, adjacent_regions in self.adjacent.items():
                for adjacent_region in adjacent_regions:
                    f.write(f"adjacent({region}, {adjacent_region}).\n")
        