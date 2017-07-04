pschitt! is a software to image an object in the sky by an array of ground Telescopes.

It has been developed to run simple tests on Cherenkov reconstruction analysis.
Up to now, the simulated objects are very simples - the most elegant one is an ellipsoid representing an electromagnetic shower.

Copyright (C) 2016  Thomas Vuillaume

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

The author may be contacted @
thomas.vuillaume@lapp.in2p3.fr




Developments to be done:

- handle the different camera sizes

- object shape closer to a shower's one

- particles emitting in a cone = visible only if the telescope is this cone: compute this cone for each particle, determine if the telescope is in the cone (is_particle_visible)

- transmission coefficient in the air for each particle as a function of the distance to the telescope. first step: boolean, particle visible if transmission coefficient above a limit
