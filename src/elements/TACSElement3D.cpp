/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2014 Georgia Tech Research Corporation

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#include "TACSElement3D.h"

TACSElement3D::TACSElement3D( TACSElementModel *_model,
                              TACSElementBasis *_basis ){
  model = _model;  model->incref();
  basis = _basis;  basis->incref();
}

TACSElement3D::~TACSElement3D(){
  model->decref();
  basis->decref();
}

// Get the layout properties of the element
int TACSElement3D::getVarsPerNode(){
  return model->getVarsPerNode();
}

int TACSElement3D::getNumNodes(){
  return basis->getNumNodes();
}

int TACSElement3D::getDesignVarsPerNode(){
  return model->getDesignVarsPerNode();
}

ElementLayout TACSElement3D::getLayoutType(){
  return basis->getLayoutType();
}

TACSElementBasis* TACSElement3D::getElementBasis(){
  return basis;
}

TACSElementModel* TACSElement3D::getElementModel(){
  return model;
}

/*
  Retrieve the global design variable numbers associated with this element
*/
int TACSElement3D::getDesignVarNums( int elemIndex, int dvLen, int dvNums[] ){
  return model->getDesignVarNums(elemIndex, dvLen, dvNums);
}

/*
  Set the element design variables from the design vector
*/
int TACSElement3D::setDesignVars( int elemIndex, int dvLen,
                                  const TacsScalar dvs[] ){
  return model->setDesignVars(elemIndex, dvLen, dvs);
}

/*
  Get the element design variables values
*/
int TACSElement3D::getDesignVars( int elemIndex, int dvLen,
                                  TacsScalar dvs[] ){
  return model->getDesignVars(elemIndex, dvLen, dvs);
}

/*
  Get the lower and upper bounds for the design variable values
*/
int TACSElement3D::getDesignVarRange( int elemIndex, int dvLen,
                                      TacsScalar lb[], TacsScalar ub[] ){
  return model->getDesignVarRange(elemIndex, dvLen, lb, ub);
}

/*
  Add the residual to the provided vector
*/
void TACSElement3D::addResidual( int elemIndex,
                                 double time,
                                 const TacsScalar *Xpts,
                                 const TacsScalar *vars,
                                 const TacsScalar *dvars,
                                 const TacsScalar *ddvars,
                                 TacsScalar *res ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, dvars, ddvars,
                                              X, Xd, J, Ut, Ud, Ux);

    // Multiply the weight by the quadrature point
    detJ *= weight;

    // Evaluate the weak form of the model
    TacsScalar DUt[3*MAX_VARS_PER_NODE], DUx[3*MAX_VARS_PER_NODE];
    model->evalWeakIntegrand(elemIndex, n, time, pt, X,
                             Ut, Ux, DUt, DUx);

    // Add the weak form of the residual at this point
    basis->addWeakFormResidual(n, pt, detJ, J, vars_per_node, DUt, DUx, res);
  }
}

/*
  Add the residual and Jacobians to the arrays
*/
void TACSElement3D::addJacobian( int elemIndex,
                                 double time,
                                 TacsScalar alpha,
                                 TacsScalar beta,
                                 TacsScalar gamma,
                                 const TacsScalar *Xpts,
                                 const TacsScalar *vars,
                                 const TacsScalar *dvars,
                                 const TacsScalar *ddvars,
                                 TacsScalar *res,
                                 TacsScalar *mat ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, dvars, ddvars, X, Xd, J,
                                              Ut, Ud, Ux);

    // Multiply the weight by the quadrature point
    detJ *= weight;

    // Evaluate the weak form of the model
    int Jac_nnz;
    const int *Jac_pairs;
    TacsScalar DUt[3*MAX_VARS_PER_NODE], DUx[3*MAX_VARS_PER_NODE];
    TacsScalar Jac[36*MAX_VARS_PER_NODE*MAX_VARS_PER_NODE];
    model->evalWeakJacobian(elemIndex, n, time, pt, X, Ut, Ux, DUt, DUx,
                            &Jac_nnz, &Jac_pairs, Jac);

    // Add the weak form of the residual at this point
    basis->addWeakFormJacobian(n, pt, detJ, J, vars_per_node, DUt, DUx,
                               alpha, beta, gamma,
                               Jac_nnz, Jac_pairs, Jac, res, mat);
  }
}

// Functions for the adjoint
void TACSElement3D::addAdjResProduct( int elemIndex,
                                      double time,
                                      TacsScalar scale,
                                      const TacsScalar psi[],
                                      const TacsScalar Xpts[],
                                      const TacsScalar vars[],
                                      const TacsScalar dvars[],
                                      const TacsScalar ddvars[],
                                      int dvLen,
                                      TacsScalar dfdx[] ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar Psi[MAX_VARS_PER_NODE];
    TacsScalar Psid[3*MAX_VARS_PER_NODE], Psix[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, dvars, ddvars, psi,
                                              X, Xd, J, Ut, Ud, Ux,
                                              Psi, Psid, Psix);

    // Compute the weight
    TacsScalar s = scale*detJ*weight;

    model->addWeakAdjProduct(elemIndex, time, s, n, pt, X,
                             Ut, Ux, Psi, Psix, dvLen, dfdx);
  }
}

void TACSElement3D::addAdjResXptProduct( int elemIndex,
                                         double time,
                                         TacsScalar scale,
                                         const TacsScalar psi[],
                                         const TacsScalar Xpts[],
                                         const TacsScalar vars[],
                                         const TacsScalar dvars[],
                                         const TacsScalar ddvars[],
                                         TacsScalar dfdXpts[] ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar Psi[MAX_VARS_PER_NODE];
    TacsScalar Psid[3*MAX_VARS_PER_NODE], Psix[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, dvars, ddvars, psi,
                                              X, Xd, J, Ut, Ud, Ux,
                                              Psi, Psid, Psix);

    TacsScalar product;
    TacsScalar dfdX[3], dfdUx[3*MAX_VARS_PER_NODE], dfdPsix[3*MAX_VARS_PER_NODE];
    model->evalWeakAdjXptSensProduct(elemIndex, n, time, pt, X,
                                     Ut, Ux, Psi, Psix, &product,
                                     dfdX, dfdUx, dfdPsix);

    // Scale the derivatives appropriately
    TacsScalar s = scale*detJ*weight;
    dfdX[0] *= s;  dfdX[1] *= s;  dfdX[2] *= s;

    for ( int i = 0; i < 3*vars_per_node; i++ ){
      dfdUx[i] *= s;
      dfdPsix[i] *= s;
    }

    // Compute the derivative of the determinant of the transformation
    TacsScalar dfddetJ = scale*weight*product;

    basis->addFieldGradientXptSens(pt, Xpts, vars_per_node, Xd, J, Ud, Psid,
                                   dfddetJ, dfdX, NULL, NULL, dfdUx, dfdPsix,
                                   dfdXpts);
  }
}

/**
  Compute a specific type of element matrix (mass, stiffness, geometric
  stiffness, etc.)
*/
void TACSElement3D::getMatType( ElementMatrixType matType,
                                int elemIndex,
                                double time,
                                const TacsScalar Xpts[],
                                const TacsScalar vars[],
                                TacsScalar mat[] ){
  // Zero the element matrix
  const int nvars = getNumVariables();
  memset(mat, 0, nvars*nvars*sizeof(TacsScalar));

  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, NULL, NULL, X, Xd, J,
                                              Ut, Ud, Ux);

    // Multiply the weight by the quadrature weight
    detJ *= weight;

    // Evaluate the weak form of the model
    int Jac_nnz;
    const int *Jac_pairs;
    TacsScalar Jac[36*MAX_VARS_PER_NODE*MAX_VARS_PER_NODE];
    model->evalWeakMatrix(matType, elemIndex, time, n, pt, X, Ut, Ux,
                          &Jac_nnz, &Jac_pairs, Jac);

    // Add the weak form of the residual at this point
    double alpha = 1.0, beta = 1.0, gamma = 1.0;
    basis->addWeakFormJacobian(n, pt, detJ, J, vars_per_node,
                               NULL, NULL, alpha, beta, gamma,
                               Jac_nnz, Jac_pairs, Jac, NULL, mat);
  }
}

/**
  Add the derivative of the product of a specific matrix w.r.t.
  the design variables
*/
void TACSElement3D::addMatDVSensInnerProduct( ElementMatrixType matType,
                                              int elemIndex,
                                              double time,
                                              TacsScalar scale,
                                              const TacsScalar psi[],
                                              const TacsScalar phi[],
                                              const TacsScalar Xpts[],
                                              const TacsScalar vars[],
                                              int dvLen,
                                              TacsScalar dfdx[] ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar Psit[3*MAX_VARS_PER_NODE];
    TacsScalar Psid[3*MAX_VARS_PER_NODE], Psix[3*MAX_VARS_PER_NODE];
    TacsScalar Phit[3*MAX_VARS_PER_NODE];
    TacsScalar Phid[3*MAX_VARS_PER_NODE], Phix[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, NULL, NULL, X, Xd, J,
                                              Ut, Ud, Ux);
    basis->getFieldGradient(pt, Xpts, vars_per_node,
                            psi, NULL, NULL, X, Xd, J,
                            Psit, Psid, Psix);
    basis->getFieldGradient(pt, Xpts, vars_per_node,
                            phi, NULL, NULL, X, Xd, J,
                            Phit, Phid, Phix);

    // Multiply by the quadrature weight
    TacsScalar s = scale*weight*detJ;

    model->addWeakMatDVSens(matType, elemIndex, time, s, n, pt, X, Ut, Ux,
                            Psit, Psix, Phit, Phix, dvLen, dfdx);
  }
}

/**
  Compute the derivative of the product of a specific matrix w.r.t.
  the input variables (vars).
*/
void TACSElement3D::getMatSVSensInnerProduct( ElementMatrixType matType,
                                              int elemIndex,
                                              double time,
                                              const TacsScalar psi[],
                                              const TacsScalar phi[],
                                              const TacsScalar Xpts[],
                                              const TacsScalar vars[],
                                              TacsScalar dfdu[] ){
  // Compute the number of quadrature points
  const int nquad = basis->getNumQuadraturePoints();
  const int vars_per_node = model->getVarsPerNode();

  // Loop over each quadrature point and add the residual contribution
  for ( int n = 0; n < nquad; n++ ){
    // Get the quadrature weight
    double pt[3];
    double weight = basis->getQuadraturePoint(n, pt);

    // Get the solution field and the solution field gradient and the
    // Jacobian transformation
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    TacsScalar Psit[3*MAX_VARS_PER_NODE];
    TacsScalar Psid[3*MAX_VARS_PER_NODE], Psix[3*MAX_VARS_PER_NODE];
    TacsScalar Phit[3*MAX_VARS_PER_NODE];
    TacsScalar Phid[3*MAX_VARS_PER_NODE], Phix[3*MAX_VARS_PER_NODE];
    TacsScalar detJ = basis->getFieldGradient(pt, Xpts, vars_per_node,
                                              vars, NULL, NULL, X, Xd, J,
                                              Ut, Ud, Ux);
    basis->getFieldGradient(pt, Xpts, vars_per_node,
                            psi, NULL, NULL, X, Xd, J,
                            Psit, Psid, Psix);
    basis->getFieldGradient(pt, Xpts, vars_per_node,
                            phi, NULL, NULL, X, Xd, J,
                            Phit, Phid, Phix);

    // Multiply by the quadrature weight
    TacsScalar scale = weight*detJ;

    TacsScalar dfdUt[3*MAX_VARS_PER_NODE], dfdUx[3*MAX_VARS_PER_NODE];
    model->evalWeakMatSVSens(matType, elemIndex, time, scale, n, pt, X, Ut, Ux,
                             Psit, Psix, Phit, Phix, dfdUt, dfdUx);

    // Add the field gradient
    basis->addFieldGradientSVSens(pt, Xpts, vars_per_node, Xd, J, Ud,
                                  dfdUt, dfdUx, dfdu);
  }
}

/**
   Evaluate a point-wise quantity of interest.
*/
int TACSElement3D::evalPointQuantity( int elemIndex,
                                      int quantityType,
                                      double time,
                                      int n, double pt[],
                                      const TacsScalar Xpts[],
                                      const TacsScalar vars[],
                                      const TacsScalar dvars[],
                                      const TacsScalar ddvars[],
                                      TacsScalar *quantity ){
  const int vars_per_node = model->getVarsPerNode();
  TacsScalar X[3], Xd[9], J[9];
  TacsScalar Ut[3*MAX_VARS_PER_NODE];
  TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
  basis->getFieldGradient(pt, Xpts, vars_per_node, vars, dvars, ddvars,
                          X, Xd, J, Ut, Ud, Ux);

  return model->evalPointQuantity(elemIndex, quantityType, time, n, pt,
                                  X, Xd, Ut, Ux, quantity);
}

/**
   Add the derivative of the point quantity w.r.t. the design variables
*/
void TACSElement3D::addPointQuantityDVSens( int elemIndex,
                                            int quantityType,
                                            double time,
                                            TacsScalar scale,
                                            int n, double pt[],
                                            const TacsScalar Xpts[],
                                            const TacsScalar vars[],
                                            const TacsScalar dvars[],
                                            const TacsScalar ddvars[],
                                            const TacsScalar dfdq[],
                                            int dvLen, TacsScalar dfdx[] ){
  const int vars_per_node = model->getVarsPerNode();
  TacsScalar X[3], Xd[9], J[9];
  TacsScalar Ut[3*MAX_VARS_PER_NODE];
  TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
  basis->getFieldGradient(pt, Xpts, vars_per_node, vars, dvars, ddvars,
                          X, Xd, J, Ut, Ud, Ux);

  model->addPointQuantityDVSens(elemIndex, quantityType, time, scale, n, pt,
                                X, Xd, Ut, Ux, dfdq, dvLen, dfdx);
}

/**
   Add the derivative of the point quantity w.r.t. the state variables
*/
void TACSElement3D::addPointQuantitySVSens( int elemIndex,
                                            int quantityType,
                                            double time,
                                            TacsScalar alpha,
                                            TacsScalar beta,
                                            TacsScalar gamma,
                                            int n, double pt[],
                                            const TacsScalar Xpts[],
                                            const TacsScalar vars[],
                                            const TacsScalar dvars[],
                                            const TacsScalar ddvars[],
                                            const TacsScalar dfdq[],
                                            TacsScalar dfdu[] ){
  const int vars_per_node = model->getVarsPerNode();
  TacsScalar X[3], Xd[9], J[9];
  TacsScalar Ut[3*MAX_VARS_PER_NODE];
  TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
  basis->getFieldGradient(pt, Xpts, vars_per_node, vars, dvars, ddvars,
                          X, Xd, J, Ut, Ud, Ux);

  // Evaluate the derivative of the function with respect to X, Ut, Ux
  TacsScalar dfdX[3], dfdXd[9];
  TacsScalar dfdUt[3*MAX_VARS_PER_NODE], dfdUx[3*MAX_VARS_PER_NODE];
  model->evalPointQuantitySens(elemIndex, quantityType, time, n, pt,
                               X, Xd, Ut, Ux, dfdq, dfdX, dfdXd, dfdUt, dfdUx);

  basis->addFieldGradientSVSens(pt, Xpts, vars_per_node, Xd, J, Ud,
                                dfdUt, dfdUx, dfdu);
}

/**
   Add the derivative of the point quantity w.r.t. the node locations
*/
void TACSElement3D::addPointQuantityXptSens( int elemIndex,
                                             int quantityType,
                                             double time,
                                             TacsScalar scale,
                                             int n, double pt[],
                                             const TacsScalar Xpts[],
                                             const TacsScalar vars[],
                                             const TacsScalar dvars[],
                                             const TacsScalar ddvars[],
                                             const TacsScalar dfdq[],
                                             TacsScalar dfdXpts[] ){
  const int vars_per_node = model->getVarsPerNode();
  TacsScalar X[3], Xd[9], J[9];
  TacsScalar Ut[3*MAX_VARS_PER_NODE];
  TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
  basis->getFieldGradient(pt, Xpts, vars_per_node, vars, dvars, ddvars,
                          X, Xd, J, Ut, Ud, Ux);

  // Evaluate the derivative of the function with respect to X, Ut, Ux
  TacsScalar dfdX[3], dfdXd[9];
  TacsScalar dfdUt[3*MAX_VARS_PER_NODE], dfdUx[2*MAX_VARS_PER_NODE];
  model->evalPointQuantitySens(elemIndex, quantityType, time, n, pt,
                               X, Xd, Ut, Ux, dfdq, dfdX, dfdXd, dfdUt, dfdUx);

  basis->addFieldGradientXptSens(pt, Xpts, vars_per_node, Xd, J, Ud,
                                 0.0, dfdX, dfdXd, NULL, dfdUx, dfdXpts);
}

/*
  Get the element data for the basis
*/
void TACSElement3D::getOutputData( int elemIndex,
                                   ElementType etype,
                                   int write_flag,
                                   const TacsScalar Xpts[],
                                   const TacsScalar vars[],
                                   const TacsScalar dvars[],
                                   const TacsScalar ddvars[],
                                   int ld_data,
                                   TacsScalar *data ){
  int num_vis_nodes = TacsGetNumVisNodes(basis->getLayoutType());
  const int vars_per_node = model->getVarsPerNode();

  // Write out the output data
  for ( int n = 0; n < num_vis_nodes; n++ ){
    double pt[3];
    basis->getVisPoint(n, pt);

    // Get the field gradient information
    TacsScalar X[3], Xd[9], J[9];
    TacsScalar Ut[3*MAX_VARS_PER_NODE];
    TacsScalar Ud[3*MAX_VARS_PER_NODE], Ux[3*MAX_VARS_PER_NODE];
    basis->getFieldGradient(pt, Xpts, vars_per_node,
                            vars, dvars, ddvars,
                            X, Xd, J, Ut, Ud, Ux);

    // Evaluate the output from the data
    double time = 0.0;
    model->getOutputData(elemIndex, time, etype, write_flag, pt,
                         X, Ut, Ux, ld_data, data);

    data += ld_data;
  }
}
