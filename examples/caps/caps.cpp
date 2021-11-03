#include "TACSAssembler.h"
#include "TACSCreator.h"
#include "TACSToFH5.h"

#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSContinuation.h"
#include "TACSMeshLoader.h"


int main( int argc, char *argv[] ){
  MPI_Init(&argc, &argv);

  const char *filename = "nastran_CAPS2.bdf";

  // Get the rank
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  TacsScalar rho = 2700.0;
  TacsScalar specific_heat = 921.096;
  TacsScalar E = 70e9; // <-check units on this
  TacsScalar nu = 0.3;
  TacsScalar ys = 270.0e6;
  TacsScalar cte = 24.0e-6;
  TacsScalar kappa = 230.0;
  TACSMaterialProperties *props =
    new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

  TacsScalar axis[] = {1.0, 0.0, 0.0};
  TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);

  TacsScalar t = 0.025;
  int t_num = 0;
  TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t, t_num);

  // TACSElement *linear_shell = new TACSQuad9Shell(transform, con);
  TACSElement *linear_shell = new TACSQuad9NonlinearShell(transform, con);
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
  int vars_per_node = 6;
  TACSAssembler *assembler = mesh->createTACS(vars_per_node);
  assembler->incref();

  TACSSchurMat *mat = assembler->createSchurMat(); // stiffness matrix

  // // Apply loads
  // TACSBVec *f = assembler->createVec(); // loads
  // TacsScalar *force_vals;
  // int size = f->getArray(&force_vals);
  // // for ( int k = 0; k < size; k += assembler->getVarsPerNode() ){
  // //   force_vals[k] -= 0.05;
  // // }
  // for ( int k = 2; k < size; k += assembler->getVarsPerNode() ){
  //   force_vals[k] -= 0.1;
  // }
  // assembler->applyBCs(f);

  // R(u, lambda) = r(u) - lambda * f

  TACSBVec *f = assembler->createVec(); // loads
  assembler->setBCs(f);



  // Allocate the factorization
  int lev = 10000;
  double fill = 10.0;
  int reorder_schur = 1;
  TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
  pc->incref();

  // Now, set up the solver
  int gmres_iters = 15;
  int nrestart = 0; // Number of allowed restarts
  int is_flexible = 0; // Is a flexible preconditioner?

  GMRES *ksm = new GMRES(mat, pc, gmres_iters, nrestart, is_flexible);
  ksm->incref();

  // Create a print object for writing output to the screen
  int freq = 1;
  KSMPrint *ksm_print = new KSMPrintStdout("KSM", rank, freq);

  TACSContinuation *cont = new TACSContinuation(assembler, 100, 50, 6, 1e-8, 1e3, 1e-3, 1e-30, 1e-8, 1e-30);
  cont->solve_tangent(mat, pc, ksm, f, 0, 0.005, ksm_print);

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
  f5->writeToFile("nastran_CAPS.f5");

  assembler->decref();
  linear_shell->decref();
  mesh->decref();
  MPI_Finalize();



  /////////////////////////////////////////////////////////
}