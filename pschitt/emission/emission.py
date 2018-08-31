import numpy as np
from .. import geometry as geo

def transmission(distances):
    """
    Compute the transmission coefficient between 0 and 1 taking into accounts all radiative transfer effects
    1 corresponds to a complete transmission (no absorption)
    Parameters
    ----------
    distances: numpy array of atmospheric length to go through

    Returns
    -------
    numpy array of floats between 0 and 1 for all distances
    """
    return np.ones(len(distances))



def mask_transmitted_particles(tel, shower):
    """
    Compute a masking array for the photons from the shower to know if they reach the telescope
    depending on their transmission probability.
    It takes into account the position of the telescope and shower direction and impact parameter.

    Parameters
    ----------
    tel: telescope class
    shower: shower class

    Returns
    -------
    numpy array of booleans of length = len(shower.particles)
    """
    # Compute the air transmission coefficient for each particle:
    particle_distances = geo.distance_shower_camera_center(shower, tel)
    transmissions = transmission(particle_distances)

    # Compute the transmission profile relative to Cherenkov cone for each particle:
    shower_direction = geo.altaz_to_normal(shower.alt, shower.az)
    # angles = np.array([geo.angle(shower_direction, particle - tel.mirror_center) for particle in shower.particles])
    # faster implementation
    tmp = (shower.particles - tel.mirror_center)
    angles = np.arccos(np.sum(shower_direction * tmp, axis=1) / np.linalg.norm(tmp, axis=1))


    # thetas = np.array([geo.angle(tel.normal, particle - tel.mirror_center) for particle in shower.particles])

    # The resulting transmission probability is the product of both:
    p_trans = transmissions * \
              shower.particles_angular_emission_profile(angles, **shower.particles_angular_emission_profile_kwargs)

    mask_transmitted_particles = p_trans > np.random.rand(len(shower.particles))

    return mask_transmitted_particles

