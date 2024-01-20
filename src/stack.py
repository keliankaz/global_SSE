#%%
import numpy as np
from scipy.stats import gaussian_kde 
from pathlib import Path
import src
from src.catalogs import (
    JapanSlowSlipCatalog,
    MexicoSlowSlipCatalog, 
    CostaRicaSlowSlipCatalog,
    WilliamsSlowSlipCatalog,
    MichelSlowSlipCatalog,
    OkadaAlaskaSlowSlipCatalog, 
)
from src.data import (
    AllSlabs, 
    EarthquakeCatalog,
)
import copy

base_dir = Path(src.__file__).parents[1]

class Stack:
    
    slab = AllSlabs()
    _earthquake_metadata = {
            "starttime": '1990-01-01',
            "endtime": '2023-01-01',
            "latitude_range": [-90,90],
            "longitude_range": [-180,180],
            "minimum_magnitude": 3.5,      # just for loading - will use max(EARTHQUAKE_MAGNITUDE_CUTOFF, 3.5)  
        }
    _earthquakes = EarthquakeCatalog(
        filename=base_dir / "Datasets" / "Seismicity_datasets" / "global_earthquakes_35.csv",
        kwargs=_earthquake_metadata,
    )
    
    def __init__(
        self,
        slowslip_catalog = None,
        regional_catalogs = None, 
        QUICK_RUN = False,                   # run sorter bootstap resampling runs
        EARTHQUAKE_MAGNITUDE_CUTOFF = 4.0,   # minimum earthquake magntitude
        DEPTH_SUBSET = 'all',                # 'deep', 'shallow', 'all' - selected using k-means clustering
        DURATION_SUBSET = 'all',             
        SLOWSLIP_MAGNITUDE_CUTOFF = 5.5,     # minimum slow slip event magntitude
        DUPLICATE_RADIUS = 70,               # km (for the removal duplicates)
        DUPLICATE_TIME = 40,                 # days (for the removal duplicates)
        TIME_WINDOW = 15,                    # source durations (e.g. 7 background, 1 coshocks, 7 aftershock)
        SMOOTHING_BW_SOURCE_DURATION = 0.25, # source durations 
        SPACE_WINDOW_BIG = 2000,             # km for spatial stack
        DISTANCE_TO_SLAB = 20,               # km (slab perpendicular distance to Slab 2.0 model for windows)
        IMPUTE_DURATION = True,              # add durations based on best fit to magnitude/duration scaling
        REPRESENTATIVE_SSE_SIZE = 50,        # km for temporal stack
        NEARBY_EQ_DISTANCE = 20,             # distance threshold to select earthquakes that are 'near' slow slip events
    ) -> None:
        
        self.QUICK_RUN = QUICK_RUN
        self.EARTHQUAKE_MAGNITUDE_CUTOFF = EARTHQUAKE_MAGNITUDE_CUTOFF
        self.DEPTH_SUBSET = DEPTH_SUBSET
        self.DURATION_SUBSET = DURATION_SUBSET
        self.SLOWSLIP_MAGNITUDE_CUTOFF = SLOWSLIP_MAGNITUDE_CUTOFF
        self.DUPLICATE_RADIUS = DUPLICATE_RADIUS
        self.DUPLICATE_TIME = DUPLICATE_TIME
        self.TIME_WINDOW = TIME_WINDOW
        self.SMOOTHING_BW_SOURCE_DURATION = SMOOTHING_BW_SOURCE_DURATION
        self.SPACE_WINDOW_BIG = SPACE_WINDOW_BIG
        self.DISTANCE_TO_SLAB = DISTANCE_TO_SLAB
        self.IMPUTE_DURATION = IMPUTE_DURATION
        self.REPRESENTATIVE_SSE_SIZE = REPRESENTATIVE_SSE_SIZE
        self.NEARBY_EQ_DISTANCE = NEARBY_EQ_DISTANCE
        self.BACKGROUND_DURATION = (TIME_WINDOW-1)/2 # source durations 
        
        if slowslip_catalog and regional_catalogs:
            self.slowslip = slowslip_catalog
            self.regional_slowslip = regional_catalogs
        elif slowslip_catalog and not regional_catalogs:
            self.slowslip = slowslip_catalog
        elif not slowslip_catalog and regional_catalogs:
            NotImplementedError
        else:
            self.slowslip, self.regional_slowslip = self.get_slowslipevents()
            
        self.earthquakes = copy.deepcopy(self._earthquakes)
        self.earthquakes.mag_completeness = self.EARTHQUAKE_MAGNITUDE_CUTOFF

        local_earthquakes = self.earthquakes.intersection(self.slowslip,buffer_radius_km=SPACE_WINDOW_BIG)
        distance_to_slab = self.slab.distance(
            local_earthquakes.catalog[["lat","lon","depth"]].values,
            depth_unit="km",
            distance_unit="km",
        )
        self.local_earthquakes = EarthquakeCatalog(local_earthquakes.catalog.loc[distance_to_slab < self.DISTANCE_TO_SLAB])
        
        # depends on previous config and catalogs
        self.times_by_window, _, self.time_weights_by_window, self.indicies_by_window = self.get_windows()
        self.times, self.time_weights, self.indices = [
            np.concatenate(list_of_arrays) for list_of_arrays in [
                self.times_by_window, self.time_weights_by_window, self.indicies_by_window
            ]
        ]
        

    def get_slowslipevents(self):
        # Consider depth subsets THIS CURRENTLY RELOADS THE DATA!!
        if self.DEPTH_SUBSET == 'shallow':
            shallow_slowslip = [
                JapanSlowSlipCatalog().get_ryukyu_trench().get_clusters('depth',2)[0],
                JapanSlowSlipCatalog().get_japan_trench().get_clusters('depth',2)[0],
                JapanSlowSlipCatalog().get_boso_peninsula().get_clusters('depth',2)[0],
                CostaRicaSlowSlipCatalog().get_clusters('depth',2)[0],
                WilliamsSlowSlipCatalog().get_clusters('depth',2)[0],
            ]

            slowslip = shallow_slowslip

        elif self.DEPTH_SUBSET == 'deep':
            deep_slowslip = [
                JapanSlowSlipCatalog().get_nankai_trough().get_clusters('depth',2)[1],
                JapanSlowSlipCatalog().get_ryukyu_trench().get_clusters('depth',2)[1],
                JapanSlowSlipCatalog().get_japan_trench().get_clusters('depth',2)[1],
                JapanSlowSlipCatalog().get_boso_peninsula().get_clusters('depth',2)[1],
                MexicoSlowSlipCatalog(), 
                CostaRicaSlowSlipCatalog().get_clusters('depth',2)[1],
                WilliamsSlowSlipCatalog().get_clusters('depth',2)[1],
                MichelSlowSlipCatalog(),
                OkadaAlaskaSlowSlipCatalog().get_clusters('depth',2)[1],
            ]
            
            slowslip = deep_slowslip

        elif self.DEPTH_SUBSET == 'all':
            slowslip = [
                JapanSlowSlipCatalog().get_nankai_trough(),
                JapanSlowSlipCatalog().get_ryukyu_trench(),
                JapanSlowSlipCatalog().get_japan_trench(),
                JapanSlowSlipCatalog().get_boso_peninsula(),
                MexicoSlowSlipCatalog(), 
                CostaRicaSlowSlipCatalog(),
                WilliamsSlowSlipCatalog(),
                MichelSlowSlipCatalog(),
                OkadaAlaskaSlowSlipCatalog(),
            ]
                
        # Consider magnitude cutoff
        for ss in slowslip:
            ss.mag_completeness = self.SLOWSLIP_MAGNITUDE_CUTOFF # changing in place

        # Consider different duration types of SSEs
        if self.DURATION_SUBSET == 'short term':
            slowslip = [ss.get_short_term_events() for ss in slowslip]
        elif self.DURATION_SUBSET == 'long term':
            slowslip = [ss.get_long_term_events() for ss in slowslip] 

        slowslip = [ss for ss in slowslip if len(ss)>0]
        all_slowslip = sum(slowslip[1:], slowslip[0])

        # Consider unique events
        filter_kwargs = dict(
            buffer_radius_km = self.DUPLICATE_RADIUS,
            buffer_time_days = self.DUPLICATE_TIME,
            stategy='reference',
        )

        all_slowslip = all_slowslip.filter_duplicates(
            **filter_kwargs,
            ref_preference = all_slowslip.catalog.groupby('ref')["time"].agg('max').sort_values(ascending=False).index
        )

        slowslip_filtered = []
        for s in slowslip:
            slowslip_filtered.append(s.filter_duplicates(**filter_kwargs, ref_preference=s.catalog.groupby('ref')["time"].agg('max').sort_values(ascending=False).index))

        slowslip = slowslip_filtered
        slowslip = sorted(slowslip, key=lambda x: len(x), reverse=True)
        for s in slowslip:
            s.catalog['name'] = s.name 
            s.catalog['region'] = s.region
        all_slowslip = sum(slowslip[1:], slowslip[0])

        # Impute durations:
        if self.IMPUTE_DURATION is True:
            for ss in slowslip: 
                ss.impute_duration(
                    mag=all_slowslip.catalog.mag,
                    duration=all_slowslip.catalog.duration,
                )
            all_slowslip.impute_duration()
        
        return all_slowslip, slowslip
     
    def get_windows(self):
        return src.center_sequences(
        slowslipevents=self.slowslip,
        earthquakes=self.local_earthquakes,
        time_window=self.TIME_WINDOW,
        space_window=self.REPRESENTATIVE_SSE_SIZE,
        lag=0,
        slab_model=self.slab,
        concatenate_output= False,
        return_indices = True,
        use_durations = True,
        use_dimensions = True,
    )

    def raw_counts(self):
        return NotImplementedError
    
    def kde_timeseries(self, number_of_times=1000, trim = 1):
        """Calculates gaussian kde time series"""
        
        # trim to avoid leakage from edge of time domain
        time_array = np.linspace(-(trim*self.TIME_WINDOW)/2, (trim*self.TIME_WINDOW)/2, number_of_times) 
        bw = self.SMOOTHING_BW_SOURCE_DURATION/np.std(self.times)
        return time_array, gaussian_kde(self.times, bw_method=bw, weights=self.time_weights)(time_array)
        
    def average_rate_increase(self, t1=-0.5, t2=0.5):
        if t1 is None:
            t1 = -self.TIME_WINDOW/2
        if t2 is None:
            t2 = self.TIME_WINDOW/2
        assert t2>t1
        
        t, kde = self.kde_timeseries()
        
        time_bool = (t>t1) & (t<t2)
        
        return np.trapz(kde[time_bool], t[time_bool]) * (t[-1]-t[0]) / (t2-t1)
            
        
#%% 
if __name__=='__main__':
    stack = Stack()
    stack.average_rate_increase()

# %%
