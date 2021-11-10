#include "TACSAssembler.h"
#include "TACSCreator.h"
#include "TACSToFH5.h"

#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSContinuation.h"
#include "TACSMeshLoader.h"


int main( int argc, char *argv[] ){
  MPI_Init(&argc, &argv);

  // const char *filename = "axial_stiffened_panel.bdf";
  const char *filename = "curved_panel.bdf";

  // Get the rank
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  TacsScalar rho = 2700.0; // kg/m^3
  TacsScalar specific_heat = 921.096; // J / (kg K)
  TacsScalar E = 70.0e9; // Pa
  TacsScalar nu = 0.3;
  TacsScalar ys = 270e6; // Pa
  TacsScalar cte = 24.0e-6; // m / (m K)
  TacsScalar kappa = 230.0; // <- unknown
  TACSMaterialProperties *props =
    new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

  TacsScalar axis[] = {1.0, 0.0, 0.0};
  TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);

  TacsScalar t = 0.001; // m
  int t_num = 0;
  TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t, t_num);

  // TACSElement *linear_shell = new TACSQuad9Shell(transform, con);
  // TACSElement *linear_shell = new TACSQuad9NonlinearShell(transform, con);
  TACSElement *linear_shell = new TACSQuad9ThermalShell(transform, con);
  linear_shell->incref();


  // Load in the .bdf file using the TACS mesh loader
  TACSMeshLoader *mesh = new TACSMeshLoader(MPI_COMM_WORLD);
  mesh->incref();

  // Scan the file
  mesh->scanBDFFile(filename);

  // Set the skin/base/stiffener elements
  mesh->setElement(0, linear_shell);
  mesh->setElement(1, linear_shell);
  mesh->setElement(2, linear_shell);

  // Create the TACSAssembler object
  int vars_per_node = 7;
  TACSAssembler *assembler = mesh->createTACS(vars_per_node);
  assembler->incref();

  // Create matrix and vectors
  TACSBVec *ans = assembler->createVec(); // displacements and rotations
  TACSBVec *f = assembler->createVec(); // loads
  TACSBVec *res = assembler->createVec(); // The residual
  TACSSchurMat *mat = assembler->createSchurMat(); // stiffness matrix

  // Increment reference count to the matrix/vectors
  ans->incref();
  f->incref();
  res->incref();
  mat->incref();

  //////////////////////////////////////////////////////////////////

  // For applied heat flux

  // TacsScalar *force_vals;
  // int size = f->getArray(&force_vals);
  // for ( int k = 3; k < size; k += assembler->getVarsPerNode() ){
  //   force_vals[k] += 1;
  // }
  // assembler->applyBCs(f);

  //////////////////////////////////////////////////////////////////

  // // For applied temperature

  // TACSBVec *f = assembler->createVec(); // loads
  assembler->setBCs(f);

  //////////////////////////////////////////////////////////////////

  // For both

  // TacsScalar *force_vals;
  // int size = f->getArray(&force_vals);
  // for ( int k = 3; k < size; k += assembler->getVarsPerNode() ){
  //   force_vals[k] -= 1;
  // }

  // assembler->setBCs(f);


  //////////////////////////////////////////////////////////////////



  // Allocate the factorization
  int lev = 10000;
  double fill = 10.0;
  int reorder_schur = 1;
  TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
  pc->incref();

  // Assemble and factor the stiffness/Jacobian matrix. Factor the
  // Jacobian and solve the linear system for the displacements
  double alpha = 1.0, beta = 0.0, gamma = 0.0;
  assembler->assembleJacobian(alpha, beta, gamma, res, mat);
  pc->factor(); // LU factorization of stiffness matrix

  res->axpy(-1.0, f); // Compute res - f
  pc->applyFactor(res, ans);

  ans->scale(-1.0);
  assembler->setVariables(ans);

  // Output for visualization
  ElementType etype = TACS_BEAM_OR_SHELL_ELEMENT;
  int write_flag = (TACS_OUTPUT_NODES |
                    TACS_OUTPUT_CONNECTIVITY |
                    TACS_OUTPUT_DISPLACEMENTS |
                    TACS_OUTPUT_STRAINS |
                    TACS_OUTPUT_STRESSES |
                    TACS_OUTPUT_EXTRAS);
  TACSToFH5 *f5 = new TACSToFH5(assembler, etype, write_flag);
  f5->incref();
  f5->writeToFile("thermal_linear_curved.f5");

  assembler->decref();
  linear_shell->decref();
  mesh->decref();
  MPI_Finalize();
}