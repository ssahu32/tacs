#include "TACSAssembler.h"
#include "TACSCreator.h"
#include "TACSToFH5.h"

#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSMeshLoader.h"

// #include "TACSShellElement.h"
// #include "TACSElementVerification.h"
// #include "TACSElementAlgebra.h"
// #include "TACSShellTraction.h"

// #include <iostream>
// #include <fstream>
// #include <cstdio>
// using namespace std;

/*
  Create the TACSAssembler object and return the associated TACS
  creator object
*/
void createAssembler( MPI_Comm comm, int nx, int ny,
                      TACSElement *element,
                      TACSAssembler **_assembler, TACSCreator **_creator ){
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Set the number of nodes/elements on this proc
  int varsPerNode = element->getVarsPerNode();

  // Set up the creator object
  TACSCreator *creator = new TACSCreator(comm, varsPerNode);

  int order = 3;

  if (rank == 0){
    // Set the number of elements
    int nnx = (order-1)*nx + 1;
    int nny = (order-1)*ny;
    int numNodes = nnx*nny;
    int numElements = nx*ny;

    // Allocate the input arrays into the creator object
    int *ids = new int[ numElements ];
    int *ptr = new int[ numElements+1 ];
    int *conn = new int[ order*order*numElements ];

    // Set the element identifiers to all zero
    memset(ids, 0, numElements*sizeof(int));

    ptr[0] = 0;
    for ( int k = 0; k < numElements; k++ ){
      // Back out the i, j coordinates from the corresponding
      // element number
      int i = k % nx;
      int j = k / nx;

      // Set the node connectivity
      for ( int jj = 0; jj < order; jj++ ){
        for ( int ii = 0; ii < order; ii++ ){
          if (j == ny-1 && jj == order-1){
            conn[order*order*k + ii + order*jj] = ((order-1)*i + ii);
          }
          else {
            conn[order*order*k + ii + order*jj] =
              ((order-1)*i + ii) + ((order-1)*j + jj)*nnx;
          }
        }
      }

      ptr[k+1] = order*order*(k+1);
    }

    // Set the connectivity
    creator->setGlobalConnectivity(numNodes, numElements,
                                   ptr, conn, ids);
    delete [] conn;
    delete [] ptr;
    delete [] ids;

    int numBcs = 2*nny;
    int *bcNodes = new int[ numBcs ];
    int k = 0;

    for ( int j = 0; j < nny; j++ ){
      int node = j*nnx;
      bcNodes[k] = node;
      k++;

      node = nnx-1 + j*nnx;
      bcNodes[k] = node;
      k++;
    }

    // Following copied from TACSCreator
    int *bc_ptr = new int[ numBcs+1 ];

    // Since the bc_vars array is input as NULL, assume that
    // all the variables at this node are fully restrained.
    int *bc_vars = new int[ numBcs*3 + 1];
    bc_ptr[0] = 0;

    for ( int i = 0; i < numBcs; i++ ){
      bc_ptr[i+1] = bc_ptr[i];
      if (i == 0){
        // Fix u, v, w, and rotx
        for ( int j = 0; j < 4; j++ ){
          bc_vars[bc_ptr[i+1]] = j;
          bc_ptr[i+1]++;
        }
      }
      else {
        // Fix v, w, and rotx
        for ( int j = 0; j < 3; j++ ){
          bc_vars[bc_ptr[i+1]] = j+1;
          bc_ptr[i+1]++;
        }
      }
    }


    TacsScalar *bc_vals = new TacsScalar[ bc_ptr[numBcs] ];
    memset(bc_vals, 0, bc_ptr[numBcs]*sizeof(TacsScalar));

    // Set the boundary conditions
    creator->setBoundaryConditions(numBcs, bcNodes, bc_ptr, bc_vars, bc_vals);

    delete [] bcNodes;
    delete [] bc_ptr;
    delete [] bc_vars;
    delete [] bc_vals;

    // Set the node locations
    TacsScalar *Xpts = new TacsScalar[ 3*numNodes ];
    double t = 1;
    double L = 100.0;
    double R = 100.0/M_PI;
    double defect = 0.0;

    for ( int j = 0; j < nny; j++ ){
      double v = -M_PI + (2.0*M_PI*j)/nny;
      for ( int i = 0; i < nnx; i++ ){
        double u = 1.0*i/(nnx - 1);
        double theta = v + 0.25*M_PI*u + defect*sin(v)*cos(2*M_PI*u);
        double x = L*(u + defect*cos(v)*sin(2*M_PI*u));
        // double theta = v;
        // double x = L*u;

        int node = i + j*nnx;
        Xpts[3*node]   =  x;
        Xpts[3*node+1] =  R*cos(theta);
        Xpts[3*node+2] = -R*sin(theta);
      }
    }

    // Set the nodal locations
    creator->setNodes(Xpts);
    delete [] Xpts;

  }

  // Set the one element
  creator->setElements(1, &element);

  // Set the reordering type
  creator->setReorderingType(TACSAssembler::MULTICOLOR_ORDER,
                             TACSAssembler::GAUSS_SEIDEL);

  // Create TACS
  TACSAssembler *assembler = creator->createTACS();

  // Set the pointers
  *_assembler = assembler;
  *_creator = creator;
}

int main( int argc, char *argv[] ){
  MPI_Init(&argc, &argv);

  // Create the mesh loader object on MPI_COMM_WORLD. The
  // TACSAssembler object will be created on the same comm
  TACSMeshLoader *mesh = new TACSMeshLoader(MPI_COMM_WORLD);
  mesh->incref();

  // Get the rank
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  TacsScalar rho = 2700.0;
  TacsScalar specific_heat = 921.096;
  TacsScalar E = 70e3;
  TacsScalar nu = 0.3;
  TacsScalar ys = 270.0;
  TacsScalar cte = 24.0e-6;
  TacsScalar kappa = 230.0;
  TACSMaterialProperties *props =
    new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

  TacsScalar axis[] = {1.0, 0.0, 0.0};
  TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);

  TacsScalar t = 1;
  int t_num = 0;
  TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t, t_num);

  // TacsTestConstitutive(con, 0);

  TACSElement *linear_shell = new TACSQuad4Shell(transform, con);
  linear_shell->incref();

  TACSElement *quadratic_shell = new TACSQuad9Shell(transform, con);
  quadratic_shell->incref();

  int nx = 20, ny = 40;
  TACSAssembler *assembler;
  TACSCreator *creator;
  createAssembler(comm, nx, ny, quadratic_shell, &assembler, &creator);
  assembler->incref();
  creator->incref();

  // Free the creator object
  creator->decref();

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

  // Allocate the factorization
  int lev = 10000;
  double fill = 10.0;
  int reorder_schur = 1;
  TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
  pc->incref();

//////////////////////////////////////////////////////////////////////////////////////////
  TacsScalar *force_vals;
  int size = f->getArray(&force_vals);
  
  TacsScalar load = 1.0;

  // Set the alpha and beta parameters
  int M = 4;
  int N = 3; // 3*pi/L

  double L = 100.0;
  double R = 100.0/M_PI;

  double Alpha = 4.0/R;
  double Beta = 3*M_PI/L;

  // Get the node vector
  TACSBVec *X = assembler->createNodeVec();
  X->incref();
  assembler->getNodes(X);

  // Get the x,y,z locations of the nodes from TACS
  int nnodes = assembler->getNumNodes();
  TacsScalar *Xpts;
  X->getArray(&Xpts);

//   FILE * pvals_output;
//   pvals_output = fopen("pvals_output.txt", "w");

  for ( int node = 0; node < nnodes; node++ ){
      // Compute the pressure at this point in the shell
      double x = TacsRealPart(Xpts[3*node]);
      double y = -R*atan2(TacsRealPart(Xpts[3*node+2]), TacsRealPart(Xpts[3*node+1]));
      
      TacsScalar pval = load*sin(Beta*x)*sin(Alpha*y);
    //   TacsScalar pval = 1.0;
      double ynorm = Xpts[3*node+1] / R;
      double znorm = Xpts[3*node+2] / R;
      
      int varsPerNode = assembler->getVarsPerNode();
      force_vals[varsPerNode*node] = 0.0;
      force_vals[varsPerNode*node+1] = ynorm * pval;
      force_vals[varsPerNode*node+2] = znorm * pval;
      
    //   fprintf(pvals_output, "%d, %f, %f, %f, %f, %f, %f\n", node, Xpts[3*node], Xpts[3*node+1], Xpts[3*node+2], force_vals[varsPerNode*node], force_vals[varsPerNode*node+1], force_vals[varsPerNode*node+2]);

  }

//   fclose(pvals_output);
  assembler->applyBCs(f);

//////////////////////////////////////////////////////////////////////////////////////////


//   TacsScalar load = 1.0;

//   // Set the alpha and beta parameters
//   int M = 4;
//   int N = 3; // 3*pi/L

//   double L = 100.0;
//   double R = 100.0/M_PI;

//   double Alpha = 4.0/R;
//   double Beta = 3*M_PI/L;

// // Set the elements the node vector
//   TACSBVec *X = assembler->createNodeVec();
//   X->incref();
//   assembler->getNodes(X);

//   TACSAuxElements *aux = new TACSAuxElements();

//   for ( int elem = 0; elem < assembler->getNumElements(); elem++ ){
//     TacsScalar Xelem[3*9];
//     TacsScalar tr[3*9];

//     int nnodes;
//     const int *nodes;
//     assembler->getElement(elem, &nnodes, &nodes);
//     X->getValues(nnodes, nodes, Xelem);

//     for ( int node = 0; node < nnodes; node++ ){
//       // Compute the pressure at this point in the shell
//       double x = TacsRealPart(Xelem[3*node]);
//       double y = -R*atan2(TacsRealPart(Xelem[3*node+2]), TacsRealPart(Xelem[3*node+1]));

//       TacsScalar pval = load*sin(Beta*x)*sin(Alpha*y);
//       TacsScalar ynorm = Xelem[3*node+1] / R;
//       TacsScalar znorm = Xelem[3*node+2] / R;

//       tr[3*node] = 0.0;
//       tr[3*node+1] = ynorm * pval;
//       tr[3*node+2] = znorm * pval;
//     }

//     TACSElement *trac = NULL;
//     if (order == 2){
//       trac = new TACSShellTraction<6, TACSQuadLinearQuadrature, TACSShellQuadBasis>(tr);
//     }
//     else if (order == 3){
      // trac = new TACSShellTraction<6, TACSQuadQuadraticQuadrature, TACSShellQuadBasis>(tr);
//     }

//     aux->addElement(elem, trac);
//   }

//   X->decref();

//   // Set the auxiliary elements
  // assembler->setAuxElements(aux);



//////////////////////////////////////////////////////////////////////////////////////////

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
  f5->writeToFile("cyl_linear_fem.f5");
  assembler->decref();

  linear_shell->decref();
  quadratic_shell->decref();

  MPI_Finalize();
}