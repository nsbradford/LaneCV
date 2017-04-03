"""
	particlefilter.py
	3/29/2017
	Nicholas S. Bradford

"""

import numpy as np

from .model import State, LineModel
from .config import Constants


class ParticleFilterModel():

    LEARNING_RATE = 1e-2
    VAR_OFFSET = LineModel.OFFSET_RANGE / 100
    VAR_ORIENTATION = LineModel.ORIENTATION_RANGE / 100
    VISUALIZATION_SIZE = 300
    

    def __init__(self, n=1000):
        self.state_size = 2
        self.n = n
        self.particles = self.init_particles()
        self.weights = self.init_weights()
        self.state_matrix = self.init_state()
        self.state = None
        self.last_measurement = None
        self.last_img = None
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        assert self.state_matrix.shape == (self.state_size,), self.state_matrix.shape


    def init_particles(self):
        p_offset = np.random.uniform(low=LineModel.OFFSET_MIN, high=LineModel.OFFSET_MAX, 
                            size=self.n)
        p_orientation = np.random.uniform(low=LineModel.ORIENTATION_MIN, 
                            high=LineModel.ORIENTATION_MAX, size=self.n)
        return np.vstack((p_offset, p_orientation)).T

    def init_weights(self):
        """ Initialize to a uniform distribution"""
        w = np.ones((self.n,))
        w /= np.sum(w)
        return w

    def init_state(self):
        return self.calc_state()

    @staticmethod
    def choose_between_models(state, last_measurement):
        m1 = state.model1
        m2 = state.model2
        if state.model2 is not None and last_measurement is not None:
            observations = np.array([   [m1.offset, m1.orientation],
                                        [m2.offset, m2.orientation]])
            distance = ParticleFilterModel.distance(new_particles=observations, 
                                measurement=last_measurement)
            print('\t\tChoice 1 dist {0:.2f}: \toffset {1:.2f} \t orientation {2:.2f}'.format(
                                distance[0], m1.offset, m1.orientation))
            print('\t\tChoice 2 dist {0:.2f}: \toffset {1:.2f} \t orientation {2:.2f}'.format(
                                distance[1], m2.offset, m2.orientation))
            if distance[0] > distance[1]:
                m1, m2 = m2, m1
        return m1, m2

    def update_state(self, state_measurement):
        if not state_measurement:
            return self.matrix_to_state() # TODO should still update somewhat
        self.last_img = state_measurement.img
        model1, model2 = ParticleFilterModel.choose_between_models(state_measurement, 
                            self.last_measurement)
        measurement = np.array([model1.offset, model1.orientation])
        self.last_measurement = measurement
        return self.update(measurement)

    def update(self, measurement):
        """ Algorithm for Particle Filter:
            def f (S, U, Z): # S is particles, U is control, Z is measurement
                S' = empty set
                for i = 0 ... n: # each new particle
                    Sample J ~ {w} with replacement # weights of current S
                    Estimate x' ~ p(x' | U, s_j)
                    W' = p(z|x') # new particle weight is likelihood given estimate
                    S' = S' u {< x', w'>} # add new particle to set
                for i = 0 ... n: # for each particle,  normalize weights
                    W_i /= n

        """
        assert measurement.shape == (self.state_size,)
        resampled_indices = np.random.choice(a=self.n, size=self.n, replace=True, p=self.weights)
        resampled_particles = self.particles[resampled_indices, :]
        self.particles = self.apply_control(resampled_particles)
        self.weights = 1 /( 1 + ParticleFilterModel.distance(self.particles, measurement))
        self.weights /= np.sum(self.weights)
        self.state_matrix = self.calc_state()
        # print(np.std(self.particles[:, 0]), np.std(self.particles[:, 1]))
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        assert self.state_matrix.shape == (self.state_size,), self.state_matrix.shape
        self.state = self.matrix_to_state()
        return self.state


    @staticmethod
    def distance(new_particles, measurement):
        """ Squared distance. average of 2 columns, for each row.
            Normalize by the expected variance of true values.
        """
        transform = np.array([1000/LineModel.OFFSET_RANGE, 1000/LineModel.ORIENTATION_RANGE])
        normalized_particles = new_particles * transform
        normalized_measurement = measurement * transform
        distances = ((normalized_particles - normalized_measurement) ** 2)
        answer = distances.mean(axis=1)
        return answer * ParticleFilterModel.LEARNING_RATE

    def apply_control(self, resampled_particles):
        """ Apply control model (motion of the plane).
            For now, model as Gaussian noise applied to each particle,
                which prevents all particles collapsing to a single point.
        """
        noise1 = np.random.normal(0, ParticleFilterModel.VAR_OFFSET, (self.n, 1))
        noise2 = np.random.normal(0, ParticleFilterModel.VAR_ORIENTATION, (self.n, 1))
        noise = np.hstack((noise1, noise2))
        return resampled_particles + noise

    def calc_state(self):
        """ Estimate the state given the current distribution of particles;
            Return the mean position.
        """
        return np.average(a=self.particles, axis=0, weights=self.weights) # for each column

    def matrix_to_state(self):
        """ Convert the internal matrix to a state. """
        model1 = LineModel(offset=self.state_matrix[0], orientation=self.state_matrix[1],
                            height=Constants.IMG_SCALED_HEIGHT, width=Constants.IMG_SCALED_WIDTH)
        # model2 = LineModel(offset=self.state_matrix[2], orientation=self.state_matrix[3],
        #                     height=Constants.IMG_SCALED_HEIGHT, width=Constants.IMG_SCALED_WIDTH)
        return State(model1, None)
