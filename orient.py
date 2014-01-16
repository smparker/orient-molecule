#!/usr/bin/env python

import sys
import math as m
import numpy as np

class Geometry:
  names = []
  coordinates = np.array([])
  natoms = 0

  def __init__(self, names, coordinates):
    self.names = names
    self.coordinates = coordinates
    self.natoms = coordinates.shape[0]

  def out(self):
    print "%s\n" % self.natoms

    for i in range(self.natoms):
      print "%3s   %14.10f   %14.10f   %14.10f" % (self.names[i], self.coordinates[i,0], self.coordinates[i,1], self.coordinates[i,2])

def read(filename):
  f = open(filename)
  raw = f.readlines()
  f.close()

  natoms = int(raw.pop(0))
  comment = raw.pop(0)

  names = []
  coords = np.zeros([len(raw),3])

  for i in range(natoms):
    tmp = raw[i].split()
    names.append(tmp[0])
    coords[i,:] = [ float(tmp[1]), float(tmp[2]), float(tmp[3]) ]

  return Geometry(names, coords)

class Operation:
  def act(self, data):
    raise Exception("Improper use of Operation class!")

  def iscomposable(self, op):
    raise Exception("Improper use of Operation class!")

  def compose(self, op):
    raise Exception("Improper use of Operation class!")

class Translate(Operation):
  displacement = np.zeros(3)

  def __init__(self, displacement):
    self.displacement = displacement

  def __call__(self, data):
    for i in range(data.shape[0]):
      data[i,:] += self.displacement

  def iscomposable(self, op):
    return isinstance(op, Translate)

  def compose(self, trans):
    if not isinstance(trans, Translate):
      raise Exception("Improper use of Translate.compose()!")
    else:
      self.displacement += trans.displacement

class Rotate(Operation):
  A = np.eye(3)

  def __init__(self, arg1, arg2=None):
    if (arg2 is None):
      self.A = arg1
    else:
      axis = arg1
      angle = arg2
      theta = m.radians(angle)

      costheta = m.cos(theta)
      sintheta = m.sin(theta)

      Ex = np.array([ [0.0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.0] ])

      self.A = costheta * np.eye(3) + (1.0 - costheta) * np.dot(axis.reshape(3,1),axis.reshape(1,3)) + sintheta * Ex

  def __call__(self,data):
    tmp = np.dot(data, self.A.transpose())
    data[:] = tmp[:]

  def iscomposable(self, op):
    return isinstance(op, Rotate)

  def compose(self, rot):
    if not isinstance(rot, Rotate):
      raise Exception("Improper use of Rotate.compose()")
    else:
      self.A = np.dot(self.A, rot.A)

class OperationList:
  operations = []

  def append(self, op):
    if( len(self) == 0 ):
      self.operations.append(op)
    elif( self[-1].iscomposable(op) ):
      self[-1].compose(op)
    else:
      self.operations.append(op)

  def __len__(self):
    return len(self.operations)

  def __getitem__(self, key):
    return self.operations[key]

  def __iter__(self):
    return iter(self.operations)

if __name__ == '__main__':
  if (len(sys.argv) == 1) :
    print "Usage:"
    print "  %s <filename> [operations]+\n" % sys.argv[0].split('/')[-1]
    print "File must be in xyz foramt. Operations can be strung together. Allowed operations are:"
    print "    -t[xyz] <distance>             \t -- translate in x, y, or z direction"
    print "    -ta <atom>                     \t -- translate <atom> to origin"
    print "    -r[xyz] <angle>                \t -- rotate around given axis"
    print "    -rp <atom> <atom> <angle>      \t -- rotate around axis defined by pair of atoms"
    print "    -rv <x> <y> <z> <angle>        \t -- rotate around defined vector"
    print "    -a <atom1> <atom2> <atom3>     \t -- align such that atom1 and atom2 lie along the x-axis and atom3 is in the xy-plane"
    exit()

  # First, interpret input and build OperationsList
  ops = OperationList()

  options = sys.argv[:]

  options.pop(0)
  filename = options.pop(0) # filename MUST come first
  geom = read(filename)

  while( options ):
    opt = options.pop(0)
    if ( opt[1] == 't' ):
      if ( len(opt) != 3 ):
        raise Exception("Need to specify a translation option (x, y, z, a)")
      else:
        translation = np.zeros([3])
        if (opt[2] == 'x'): translation[0] = float(options.pop(0))
        elif (opt[2] == 'y'): translation[1] = float(options.pop(0))
        elif (opt[2] == 'z'): translation[2] = float(options.pop(0))
        elif (opt[2] == 'a'): translation[:] = -geom.coordinates[int(options.pop(0))-1, :]
        else: raise Exception("Unrecognized translation option")

        ops.append(Translate(translation))
    elif ( opt[1] == 'r' ):
      if ( len(opt) != 3 ):
        raise Exception("Need to specify a rotation option (x, y, z, p, v)")
      else:
        axis = np.zeros([3])
        translation = np.zeros(3) # Used if I need to translate before and after
        if (opt[2] == 'x'): axis[0] = 1.0
        elif (opt[2] == 'y'): axis[1] = 1.0
        elif (opt[2] == 'z'): axis[2] = 1.0
        elif (opt[2] == 'p'): # atom Pairs
          iatom = int(options.pop(0)) - 1
          jatom = int(options.pop(0)) - 1

          translation = 0.5 * (geom.coordinates[iatom,:] + geom.coordinates[jatom,:])

          axis = geom.coordinates[jatom, :] - geom.coordinates[iatom,:]
          axis /= np.linalg.norm(axis)
        elif (opt[2] == 'v'): # vector
          axis = np.array([float(options.pop(0)), float(options.pop(0)), float(options.pop(0))])
          axis /= np.linalg.norm(axis)
        else: raise Exception("Unrecognized rotation option")

        theta = float(options.pop(0))
        if (opt[2] == 'x' or opt[2] == 'y' or opt[2] == 'z') :
          ops.append(Rotate(axis, theta))
        elif (opt[2] == 'p' or opt[2] == 'v'):
          ops.append(Translate(-1.0*translation))
          ops.append(Rotate(axis, theta))
          ops.append(Translate(translation))
    elif ( opt[1] == 'a' ):
      # align is called with -a <atom> <atom> <atom>
      iatom = geom.coordinates[int(options.pop(0))-1,:]
      jatom = geom.coordinates[int(options.pop(0))-1,:]
      katom = geom.coordinates[int(options.pop(0))-1,:]

      ops.append(Translate(-0.5*(iatom + jatom)))

      vec1 = jatom[:] - iatom[:]
      vec1 /= np.linalg.norm(vec1)

      vec2 = katom[:] - iatom[:]
      vec2 -= np.dot(vec1,vec2) * vec1[:]
      vec2 /= np.linalg.norm(vec2)

      vec3 = np.cross(vec1,vec2)

      rotation_matrix = np.array([ vec1[:], vec2[:], vec3[:] ])
      ops.append(Rotate(rotation_matrix))
    else:
      raise Exception("Unknown operation")

  for op in ops:
    op(geom.coordinates)

  geom.out()
