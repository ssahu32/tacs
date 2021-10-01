import sys, os

import numpy as np
from mpi4py import MPI
from tacs import TACS, constitutive, elements
# sys.path.append(os.path.join(os.getenv('HOME'), 'tacs', 'examples', 'shell'))
# import shell_element
# import shell_element as elements

comm = MPI.COMM_WORLD

filename = 'axial_stiffened_panel.bdf'

# filename = 'model_mesh.bdf'
# filename = 'skewed_plate.bdf'
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile(filename)

# Create the material properties
props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)
con = constitutive.IsoShellConstitutive(props, t=0.001)

# Loop over components, creating stiffness
# and element object for each
skin_axis = [1.0, 0.0, 0.0]
skin_transform = elements.ShellRefAxisTransform(skin_axis)

num_components = mesh.getNumComponents()

# print(num_components)
for i in range(num_components):
    descriptor = mesh.getElementDescript(i)
    component = mesh.getComponentDescript(i)
    # print(i)
    # print('Descriptor:', descriptor)
    # print('Component:', component)

    # Create the linear element
    # element = shell_element.QuadLinearShell(skin_transform, con)
    element = elements.Quad9Shell(skin_transform, con)

    # Set the element into the mesh
    mesh.setElement(i, element)

# Create the assembler object
# print('element.getVarsPerNode(): ', element.getVarsPerNode())
elems = mesh.getConnectivity()

# print(elements)
assembler = mesh.createTACS(element.getVarsPerNode())

# # # Create the forces, apply them with the assembler
# forces = assembler.createVec()
# force_array = forces.getArray()
# force_array[2::6] += 100.0 # uniform load in z direction
# assembler.applyBCs(forces)

# Set up the solver
ans = assembler.createVec()
res = assembler.createVec()
update = assembler.createVec()
mat = assembler.createSchurMat()

pc = TACS.Pc(mat)
subspace = 20
restarts = 2
gmres = TACS.KSM(mat, pc, subspace, restarts)
assembler.applyBCs(ans)
assembler.setVariables(ans)

alpha = 1.0
beta = 0.0
gamma = 0.0
# assembler.zeroVariables()
# assembler.assembleJacobian(alpha, beta, gamma, None, mat)
assembler.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

# Solve the linear system and set the varaibles into TACS
# gmres.solve(forces, ans)
# gmres.solve(res, ans)
gmres.solve(res, update)
ans.axpy(-1.0, update)
# ans.scale(-1.0)
assembler.setVariables(ans)

# Find node positions
node_vec = assembler.createNodeVec()
assembler.getNodes(node_vec)
node_array = node_vec.getArray()
# print(node_array)

nodeFile = open('nodes.txt', 'w')
print(node_array, file = nodeFile)
nodeFile.close()
print(len(node_array))

etype = TACS.BEAM_OR_SHELL_ELEMENT
write_flag = (TACS.OUTPUT_NODES | TACS.OUTPUT_CONNECTIVITY |
              TACS.OUTPUT_DISPLACEMENTS | TACS.OUTPUT_STRAINS |
              TACS.OUTPUT_STRESSES | TACS.OUTPUT_EXTRAS)
f5 = TACS.ToFH5(assembler, etype, write_flag)
f5.writeToFile('axial_stiffened_panel.f5')
# f5.writeToFile('skewed_plate.f5')
