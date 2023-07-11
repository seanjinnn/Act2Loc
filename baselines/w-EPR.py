import powerlaw
from collections import defaultdict
import datetime
import math
from skmob.utils import constants, utils, gislib
from scipy.sparse import lil_matrix
import inspect
import numpy as np
from math import sqrt, sin, cos, pi, asin, pow, ceil
import geopandas as gpd
import pandas as pd
import datetime
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.models.gravity import Gravity
from od_matrix import *
from tqdm import tqdm
import pickle
import skmob
from skmob.preprocessing import filtering, compression, detection, clustering
import multiprocessing

earth_distance_km = gislib.getDistance

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID


def compute_od_matrix(gravity_singly, spatial_tessellation, tile_id_column=constants.TILE_ID,
                      relevance_column=constants.RELEVANCE):
    """
    Compute a matrix :math:`M` where element :math:`M_{ij}` is the probability p_{ij} of moving between
    locations :math:`i` and location :math:`j`, where each location refers to a row in `spatial_tessellation`.

    Parameters
    ----------
    gravity_singly : object
        instance of class `collective.Gravity` with argument `gravity_type='singly constrained'`.

    spatial_tessellation : GeoDataFrame
        the spatial tessellation describing the division of the territory in locations.

    tile_id_column : str or int, optional
        column of in `spatial_tessellation` containing the identifier of the location/tile. The default value is constants.TILE_ID.

    relevance_column : str or int, optional
        column in `spatial_tessellation` containing the relevance of the location/tile.

    Returns
    -------
    od_matrix : numpy array
        two-dimensional numpy array with the trip probabilities for each origin-destination pair.
    """
    od_matrix = gravity_singly.generate(spatial_tessellation,
                                        tile_id_column=tile_id_column,
                                        tot_outflows_column=None,
                                        relevance_column=relevance_column,
                                        out_format='probabilities')
    print(od_matrix)
    return od_matrix


def populate_od_matrix(location, lats_lngs, relevances, gravity_singly):
    """
    Populate the origin-destination matrix with the probability to move from the location in input to all other locations in the spatial tessellation.

    Parameters
    ----------
    location : int
        the identifier of a location.

    lats_lngs : list or numpy array
        list of coordinates of the centroids of the tiles in a spatial tessellation.

    relevances : list or numpy array
        list of relevances of the tiles in a spatial tessellation.

    gravity_singly : object
        instance of class `collective.Gravity` with argument `gravity_type='singly constrained'`.

    Returns
    -------
        a numpy array of trip probabilities between the origin location and each destination.
    """
    ll_origin = lats_lngs[location]
    distances = np.array([earth_distance_km(ll_origin, l) for l in lats_lngs])

    scores = gravity_singly._compute_gravity_score(distances, relevances[location, None], relevances)[0]
    return scores / sum(scores)


class WEPR():
    def __init__(self, name='WEPR', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        self._name = name

        self._rho = rho
        self._gamma = gamma
        self._beta = beta
        self._tau = tau

        self._location2return = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self.lats_lngs = None
        self.relevances = None
        self._starting_loc = None
        self.gravity_singly = None

        # Minimum waiting time (in hours)
        self._min_wait_time = min_wait_time_minutes / 60.0  # minimum waiting time

        self._t_active = 17

        self._trajectories_ = []

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def beta(self):
        return self._beta

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @property
    def t_active(self):
        return self._t_active

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation

    @property
    def trajectories_(self):
        return self._trajectories_

    def _assign_home_location(self, spatial_tessellation, n_agents):
        """
        Setting anchor point.
        In this phase,
        we simply regard random location as home that represent anchor point.

        Parameters
        ----------
        spatial_tessellation : GeoDataFrame
        the spatial tessellation describing the division of the territory in locations.

        """

        tiles = np.fromiter(spatial_tessellation['tile_ID'], dtype=int)
        pop = np.fromiter(spatial_tessellation['relevance'], dtype=int)
        prob = pop / np.sum(pop)
        home_location = np.random.choice(range(len(tiles)), size=n_agents, p=prob)
        return home_location

    def _preferential_return(self, current_location):
        """
        Select a random location given the agent's visitation frequency. Used by the return mechanism.

        Parameters
        ----------
        current_location : int
            identifier of a location.

        Returns
        -------
        int
            a location randomly chosen according to its relevance.
        """
        locations = np.fromiter(self._location2return.keys(), dtype=int)
        weights = np.fromiter(self._location2return.values(), dtype=int)

        currloc_idx = np.where(locations == current_location)
        locations = np.delete(locations, currloc_idx)
        weights = np.delete(weights, currloc_idx)

        weights = weights / np.sum(weights)
        return_location = np.random.choice(locations, p=weights, size=1)[0]
        return return_location

    def _weighted_exploration(self, current_location):
        """
        for exploration phase,there are 3 candidates take into account:
            1* Gravaty model
            2* POW model
            3* PDF model 
        """

        # Gravity model
        if self._is_sparse:
            prob_array = self._od_matrix.getrowview(current_location)
            if prob_array.nnz == 0:
                # if the row has been not populated
                weights = populate_od_matrix(current_location, self.lats_lngs, self.relevances, self.gravity_singly)
                self._od_matrix[current_location, :] = weights
            else:
                weights = prob_array.toarray()[0]
            locations = np.arange(len(self.lats_lngs))
            explore_location = np.random.choice(locations, size=1, p=weights)[0]

        else:  # if the matrix is precomputed
            locations = np.arange(len(self._od_matrix[current_location]))
            weights = self._od_matrix[current_location]
            explore_location = np.random.choice(locations, size=1, p=weights)[0]

        return explore_location

    def _get_trajdataframe(self, parameters):
        """
        Transform the trajectories list into a pandas DataFrame.

        Returns
        -------
        pandas DataFrame
            the trajectories of the agent.
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self.lats_lngs[s][0],
                                                                           longitude: self.lats_lngs[s][1]}))
        df = df.sort_values(by=[user_id, date_time]).drop('location', axis=1)
        return TrajDataFrame(df, parameters=parameters)

    def _choose_location(self):
        """
        Choose the next location to visit given the agent's current location.

        Returns
        -------
        int
            the identifier of the next location the agent has to visit.
        """
        n_visited_locations = len(self._location2return)  # number of already visited locations

        if n_visited_locations == 1:
            next_location = self._weighted_exploration(self._starting_loc)
            return next_location

        agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location

        # choose a probability to return or explore
        p_new = np.random.uniform(0, 1)

        if (p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != \
            self._od_matrix.shape[0]) or n_visited_locations == 1:  # choose to return or explore
            # weighted EXPLORATION
            next_location = self._weighted_exploration(current_location)
            # TODO: remove the part below and exclude visited locations
            #  from the list of potential destinations in _preferential_exploration
            # while next_location in self._location2return:
            #     next_location = self._preferential_exploration(current_location)
            return next_location

        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return(current_location)
            return next_location

    def _choose_waiting_time(self):
        return powerlaw.Truncated_Power_Law(xmin=self.min_wait_time, xmax=self.t_active, \
                                            parameters=[1. + self._beta, 1.0 / self._tau]).generate_random()[0]

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1, od_matrix=None,
                 relevance_column=constants.RELEVANCE, random_state=None, lats_lngs=None):

        if gravity_singly == {}:
            self.gravity_singly = Gravity(gravity_type='singly constrained')
        elif type(gravity_singly) is Gravity:
            if gravity_singly.gravity_type == 'singly constrained':
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError(
                    "Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'.")
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

        # Save function arguments and values in a dictionary
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = dict([])
        parameters['model'] = {'class': self.__class__.__init__,
                               'generate': {i: arg_values[i] for i in args[1:] if i not in ['spatial_tessellation',
                                                                                            'od_matrix', 'log_file',
                                                                                            'starting_locations']}}

        # if specified, fix the random seeds to guarantee reproducibility of simulation
        if random_state is not None:
            # random.seed(random_state)
            np.random.seed(random_state)

        # initialization of trajectories
        self._trajectories_ = []

        # setting of spatial tessellation
        num_locs = len(spatial_tessellation)
        if lats_lngs is None:
            self.lats_lngs = spatial_tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        else:
            self.lats_lngs = lats_lngs
        self.relevances = spatial_tessellation[relevance_column].fillna(0).values

        # assign individual's home location
        starting_locations = self._assign_home_location(spatial_tessellation, n_agents)

        # initialization of od matrix
        if od_matrix is None:
            self._od_matrix = lil_matrix((num_locs, num_locs))
            self._is_sparse = True
        else:
            # TODO: check it is a properly formatted stochastic matrix
            self._od_matrix = od_matrix
            self._is_sparse = False

        # for each agent
        loop = range(1, n_agents + 1)
        for agent_id in tqdm(loop):
            self._location2return = defaultdict(int)
            self._starting_loc = starting_locations[agent_id - 1]

            self._epr_generate_one_agent(agent_id, start_date, end_date)

        tdf = self._get_trajdataframe(parameters)
        return tdf

    def _epr_generate_one_agent(self, agent_id, start_date, end_date):
        total_t_active = 0
        current_date = start_date
        self._trajectories_.append((agent_id, current_date, self._starting_loc))
        self._location2return[self._starting_loc] += 1

        waiting_time = self._choose_waiting_time()
        current_date += datetime.timedelta(hours=waiting_time)
        total_t_active += waiting_time

        while current_date < end_date:
            if total_t_active <= self._t_active:
                # compare with tactive,if accumulative waiting time per day smaller:
                next_location = self._choose_location()
                self._trajectories_.append((agent_id, current_date, next_location))
                self._location2return[next_location] += 1

                waiting_time = self._choose_waiting_time()
                current_date += datetime.timedelta(hours=waiting_time)
                total_t_active += waiting_time
            else:
                # if accumulative waiting time per day overstep:

                # current_date += datetime.timedelta(hours=7)
                # self._location2return[self._starting_loc] += 1
                # self._trajectories_.append((agent_id,current_date,self._starting_loc))
                # total_t_active = 0

                next_location = self._starting_loc
                self._trajectories_.append((agent_id, current_date, next_location))

                current_date = current_date - datetime.timedelta(hours=total_t_active) + datetime.timedelta(hours=24)
                self._trajectories_.append((agent_id, current_date, next_location))
                self._location2return[next_location] += 1
                total_t_active = 0

                waiting_time = self._choose_waiting_time()
                current_date += datetime.timedelta(hours=waiting_time)
                total_t_active += waiting_time

def run_wepr(value):

    # load the spatial tessellation
    tessellation = gpd.read_file(r'C:\Users\86152\Desktop\Home\tessellation\shenzhen\sz_1km\sz_1km.shp')
    spatial_tessellation, M = load_spatial_tessellation(tessellation)
    rank_matrix = rank_od_matrix(spatial_tessellation)

    tessellation = tessellation.rename(columns={'pop': 'relevance'})
    start_date = pd.to_datetime('2021-11-01 00:00:00')
    end_date = pd.to_datetime('2021-11-14 23:59:59')
    wepr = WEPR()

    tdf = wepr.generate(start_date, end_date,  tessellation, od_matrix=rank_matrix, n_agents=200000)
    tdf = skmob.TrajDataFrame(tdf)
    tdf = compression.compress(tdf)

    pickled = open(r"C:\Users\86152\轨迹生成\dataset\wepr\wepr_1km"+str(value)+".pkl", 'wb')
    pickle.dump(tdf, pickled)

if __name__ == '__main__':

    pool_obj = multiprocessing.Pool()
    pool_obj.map(run_wepr, range(1, 50))



