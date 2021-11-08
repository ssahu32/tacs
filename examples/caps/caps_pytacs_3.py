"""
This wingbox is a simplified version of the one of the University of Michigan uCRM-9.
We use a couple of pyTACS load generating methods to model various wing loads under cruise.
The script runs the structural analysis, evaluates the wing mass and von misses failure index
and computes sensitivities with respect to wingbox thicknesses and node xyz locations.
"""
# ==============================================================================
# Standard Python modules
# ==============================================================================
from __future__ import print_function
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint
import numpy as np
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import TACS, functions, constitutive, elements, pyTACS, problems

comm = MPI.COMM_WORLD

# Instantiate FEASolver
structOptions = {
    'printtiming':True,
}

bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS.dat')
FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

# Material properties
rho = 2780.0        # density kg/m^3
E = 73.1e9          # Young's modulus (Pa)
nu = 0.33           # Poisson's ratio
kcorr = 5.0/6.0     # shear correction factor
ys = 324.0e6        # yield stress

# Shell thickness
t = 0.01            # m
tMin = 0.002        # m
tMax = 0.05         # m

# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

    # # Define reference axis for local shell stresses
    # if 'SKIN' in compDescript: # USKIN + LSKIN
    #     sweep = 35.0 / 180.0 * np.pi
    #     refAxis = np.array([np.sin(sweep), np.cos(sweep), 0])
    # else: # RIBS + SPARS + ENGINE_MOUNT
    #     refAxis = np.array([0.0, 0.0, 1.0])

    refAxis = np.array([1.0, 0.0, 0.0])

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    transform = elements.ShellRefAxisTransform(refAxis)
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            elem = elements.Quad4Shell(transform, con)
        elif elemDescript in ['CTRIA3', 'CTRIAR']:
            elem = elements.Tri3Shell(transform, con)
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        elemList.append(elem)

    # Add scale for thickness dv
    scale = [100.0]
    return elemList, scale

print("line 80")

# Set up elements and TACS assembler
# assembler = FEASolver.assembler
print("line 84")
FEASolver.createTACSAssembler(elemCallBack)
print("line 85")
assembler = FEASolver.assembler
print("line 88")

# Add TACS Functions
# FEASolver.addFunction('wing_mass', functions.StructuralMass)
# FEASolver.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.5,
#                       ksWeight=100.0)

# ==============================================================================
# Setup static problem
# ==============================================================================
# Static problem
# evalFuncs = ['wing_mass', 'ks_vmfailure']
# sp = problems.StaticProblem(name='wing', evalFuncs=evalFuncs, loadFactor=2.5)

print("line 97")

# Various methods for adding loads to structural problem:
# Let's model an engine load of 75kN weight, and 64kN thrust attached to spar
# compIDs = FEASolver.selectCompIDs(["WING_SPARS/LE_SPAR/SEG.16", "WING_SPARS/LE_SPAR/SEG.17"])
# We = 75.0e3 # N
# Te = 64.0e3 # N
# FEASolver.addLoadToComponents(sp, compIDs, [-Te, 0.0, -We, 0.0, 0.0, 0.0], averageLoad=True)
# # Next we'll approximate aerodynamic loads on upper/lower skin with a uniform traction
# L = 3e3 # N/m^2
# D = 150 # N/m^2
# tracVec = np.array([D, 0.0, L])
# compIDs = FEASolver.selectCompIDs(include='SKIN')
# FEASolver.addTractionToComponents(sp, compIDs, tracVec)
# # Finally, we can approximate fuel load by adding a pressure load to the lower skin
# P = 2e3 # N/m^2
# compIDs = FEASolver.selectCompIDs(include='L_SKIN')
# FEASolver.addPressureToComponents(sp, compIDs, P)

# Solve structural problem
# FEASolver(sp)

# Evaluate functions
# funcs = {}
# FEASolver.evalFunctions(sp, funcs)
# if comm.rank == 0:
#     pprint(funcs)

# # Solve adjoints and evaluate function sensitivities
# funcsSens = {}
# FEASolver.evalFunctionsSens(sp, funcsSens)
# if comm.rank == 0:
#     pprint(funcsSens)

# FEASolver.writeSolution(outputDir=os.path.dirname(__file__))

print("line 133")

########################################################################################

# # Create tacs assembler object from mesh loader
# # tacs = struct_mesh.createTACS(6)

# # Create the KS Function
# ksWeight = 100.0
# funcs = [functions.KSFailure(tacs, ksWeight=ksWeight)]
# # funcs = [functions.StructuralMass(tacs)]
# # funcs = [functions.Compliance(tacs)]

# print("line 146")

# # Get the design variable values
# x = tacs.createDesignVec()
# x_array = x.getArray()
# tacs.getDesignVars(x)

# print("line 153")

# # Get the node locations
# X = tacs.createNodeVec()
# tacs.getNodes(X)
# tacs.setNodes(X)

# # Create the forces
# forces = tacs.createVec()
# force_array = forces.getArray() 
# force_array[2::6] += 100.0 # uniform load in z direction
# tacs.applyBCs(forces)

# print("line 162")

# # Set up and solve the analysis problem
# res = tacs.createVec()
# ans = tacs.createVec()
# u = tacs.createVec()
# mat = tacs.createSchurMat()
# pc = TACS.Pc(mat)
# subspace = 100
# restarts = 2
# gmres = TACS.KSM(mat, pc, subspace, restarts)

# # Assemble the Jacobian and factor
# alpha = 1.0
# beta = 0.0
# gamma = 0.0
# tacs.zeroVariables()
# tacs.assembleJacobian(alpha, beta, gamma, res, mat)
# pc.factor()

# # Solve the linear system
# gmres.solve(forces, ans)
# tacs.setVariables(ans)

# # Evaluate the function
# fvals1 = tacs.evalFunctions(funcs)

# # Solve for the adjoint variables
# adjoint = tacs.createVec()
# res.zeroEntries()
# tacs.addSVSens([funcs[0]], [res])
# gmres.solve(res, adjoint)

# # Compute the total derivative w.r.t. material design variables
# fdv_sens = tacs.createDesignVec()
# fdv_sens_array = fdv_sens.getArray()
# tacs.addDVSens([funcs[0]], [fdv_sens])
# tacs.addAdjointResProducts([adjoint], [fdv_sens], -1)
# # Finalize sensitivity arrays across all procs
# fdv_sens.beginSetValues()
# fdv_sens.endSetValues()

# # Create a random direction along which to perturb the nodes
# pert = tacs.createNodeVec()
# X_array = X.getArray()
# pert_array = pert.getArray()
# pert_array[0::3] = X_array[1::3]
# pert_array[1::3] = X_array[0::3]
# pert_array[2::3] = X_array[2::3]

# # Compute the total derivative w.r.t. nodal locations
# fXptSens = tacs.createNodeVec()
# tacs.addXptSens([funcs[0]], [fXptSens])
# tacs.addAdjointResXptSensProducts([adjoint], [fXptSens], -1)
# # Finalize sensitivity arrays across all procs
# fXptSens.beginSetValues()
# fXptSens.endSetValues()

# # Set the complex step
# xpert = tacs.createDesignVec()
# xpert.setRand()
# xpert_array = xpert.getArray()
# xnew = tacs.createDesignVec()
# xnew.copyValues(x)
# if TACS.dtype is complex:
#     dh = 1e-30
#     xnew.axpy(dh*1j, xpert)
# else:
#     dh = 1e-6
#     xnew.axpy(dh, xpert)

# # Set the design variables
# tacs.setDesignVars(xnew)

# # Compute the perturbed solution
# tacs.zeroVariables()
# tacs.assembleJacobian(alpha, beta, gamma, res, mat)
# pc.factor()
# gmres.solve(forces, u)
# tacs.setVariables(u)

# # Evaluate the function for perturbed solution
# fvals2 = tacs.evalFunctions(funcs)

# if TACS.dtype is complex:
#     fd = fvals2.imag/dh
# else:
#     fd = (fvals2 - fvals1)/dh

# result = xpert.dot(fdv_sens)
# if tacs_comm.rank == 0:
#     print('FD:      ', fd[0])
#     print('Result:  ', result)
#     print('Rel err: ', (result - fd[0])/result)

# # Reset the old variable values
# tacs.setDesignVars(x)

# if TACS.dtype is complex:
#     dh = 1e-30
#     X.axpy(dh*1j, pert)
# else:
#     dh = 1e-6
#     X.axpy(dh, pert)

# # Set the perturbed node locations
# tacs.setNodes(X)

# # Compute the perturbed solution
# tacs.zeroVariables()
# tacs.assembleJacobian(alpha, beta, gamma, res, mat)
# pc.factor()
# gmres.solve(forces, u)
# tacs.setVariables(u)

# # Evaluate the function again
# fvals2 = tacs.evalFunctions(funcs)

# if TACS.dtype is complex:
#     fd = fvals2.imag/dh
# else:
#     fd = (fvals2 - fvals1)/dh

# # Compute the projected derivative
# result = pert.dot(fXptSens)

# if tacs_comm.rank == 0:
#     print('FD:      ', fd[0])
#     print('Result:  ', result)
#     print('Rel err: ', (result - fd[0])/result)

# # Output for visualization 
# flag = (TACS.OUTPUT_NODES |
#         TACS.OUTPUT_DISPLACEMENTS |
#         TACS.OUTPUT_STRAINS |
#         TACS.OUTPUT_EXTRAS)
# f5 = TACS.ToFH5(tacs, TACS.BEAM_OR_SHELL_ELEMENT, flag)
# f5.writeToFile('caps3_output.f5')




############################################################################################


# Create the stiffness object
props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)
stiff = constitutive.PlaneStressConstitutive(props)

# Set up the basis function
model = elements.LinearThermoelasticity2D(stiff)
basis = elements.QuadraticTriangleBasis()
elem = elements.Element2D(model, basis)

# Allocate the TACSCreator object
varsPerNode = model.getVarsPerNode()
creator = TACS.Creator(comm, varsPerNode)

# Create the tacs assembler object
# assembler = creator.createTACS()

res = assembler.createVec()
ans = assembler.createVec()
mat = assembler.createSchurMat()

# Create the preconditioner for the corresponding matrix
pc = TACS.Pc(mat)

# Assemble the Jacobian
alpha = 1.0
beta = 0.0
gamma = 0.0
assembler.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

res.getArray()[:] = 1.0
assembler.applyBCs(res)
pc.applyFactor(res, ans)
ans.scale(-1.0)

assembler.setVariables(ans)

# Create the function list
funcs = []

# Create the KS function
ksweight = 100.0
for i in range(1):
    funcs.append(functions.KSFailure(assembler, ksWeight=ksweight))

func_vals = assembler.evalFunctions(funcs)

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS |
        TACS.OUTPUT_STRESSES |
        TACS.OUTPUT_EXTRAS)
f5 = TACS.ToFH5(assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
f5.writeToFile('caps3_output.f5')