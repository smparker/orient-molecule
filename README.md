Orient molecule
===============

Provides a python script to perform routine manipulations of .xyz coordinate files used in quantum chemistry.
This script started out as Python practice but ended up being quite useful.

# Usage #
Give the script a filename and a list of operations and it will operate on the file and print the results to the screen.
For example

    orient.py water.xyz -tx 2.0 -rz 90.0 -ty -0.2

will transform the molecule found in `water.xyz` by
  - translating by 2.0 Angstroms in the _x_ direction
  - rotating 90 degrees about the _z_ axis
  - translating by -0.2 Angstrom in the _y_ direction
  
All operations are composed before applying to the molecule when possible, so the total operation count is kept down.

# Capabilities #
  - **translations**
    - by any specified vector
    - center of mass to origin
  - **rotations**
    - about a cartesian axis
    - about a user defined vector
    - about the vector defined by a pair of atoms
  - **compound transformations**:
    - orient:
      - translate center of mass to origin
      - principle axes along cartesian axes
    - align: given three atoms
      - translate midpoint of atoms 1 and 2 to origin
      - rotate so atoms 1 and 2 lie along _x_ axis, and all three atoms define _xy_ plane
      
# Requirements #
- numpy
