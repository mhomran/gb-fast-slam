import numpy as np
import random
import copy

class FastSlam:
    def __init__(self, particles):
        self.particles = particles
        
    def fast_slam(self, odom, sensor):
        '''Executes one iteration of the prediction-correction-resampling loop of FastSLAM.

        Returns the plot objects to be drawn for the current frame.
        '''

        # Perform the prediction step of the particle filter
        for particle in self.particles:
            particle.motion_update(odom)

        # Perform the correction step of the particle filter
        for particle in self.particles:
            particle.sensor_update(sensor)

        # Resample the particle set
        # Use the "number of effective particles" approach to resample only when
        # necessary. This approach reduces the risk of particle depletion.
        # For details, see Section IV.B of
        # http://www2.informatik.uni-freiburg.de/~burgard/postscripts/grisetti05icra.pdf
        # s = sum([particle.weight for particle in self.particles])
        # neff = 1. / sum([(particle.weight/s) ** 2 for particle in self.particles])
        # if neff < len(self.particles) / 2.:
        #     print ("resample")
        #     self.particles = self.resample(self.particles)

    def resample(particles):
        """Resample the set of particles.

        A particle has a probability proportional to its weight to get
        selected. A good option for such a resampling method is the so-called low
        variance sampling, Probabilistic Robotics page 109"""
        num_particles = len(particles)
        new_particles = []
        weights = [particle.weight for particle in particles]

        # normalize the weight
        sum_weights = sum(weights)
        weights = [weight / sum_weights for weight in weights]

        # the cumulative sum
        cumulative_weights = np.cumsum(weights)
        normalized_weights_sum = cumulative_weights[len(cumulative_weights) - 1]

        # check: the normalized weights sum should be 1 now (up to float representation errors)
        assert abs(normalized_weights_sum - 1.0) < 1e-5

        # initialize the step and the current position on the roulette wheel
        step = normalized_weights_sum / num_particles
        position = random.uniform(0, normalized_weights_sum)
        idx = 1

        # walk along the wheel to select the particles
        for i in range(1, num_particles + 1):
            position += step
            if position > normalized_weights_sum:
                position -= normalized_weights_sum
                idx = 1
            while position > cumulative_weights[idx - 1]:
                idx = idx + 1

            new_particles.append(copy.deepcopy(particles[idx - 1]))
            new_particles[i - 1].weight = 1 / num_particles

        return new_particles