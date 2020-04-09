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

#include "TACSFH5Loader.h"
#include "TacsUtilities.h"
#include "TACSMarchingCubes.h"

TACSFH5Loader::TACSFH5Loader(){
  data_file = NULL;

  comp_nums = NULL;
  ltypes = NULL;
  ptr = NULL;
  conn = NULL;

  num_elements = -1;
  conn_size = -1;

  continuous_data = NULL;
  continuous_zone = NULL;
  continuous_vars = NULL;
  num_nodes_continuous = -1;
  num_vals_continuous = -1;

  element_data = NULL;
  element_zone = NULL;
  element_vars = NULL;
  num_nodes_element = -1;
  num_vals_element = -1;
}

TACSFH5Loader::~TACSFH5Loader(){
  if (comp_nums){ delete [] comp_nums; }
  if (ltypes){ delete [] ltypes; }
  if (ptr){ delete [] ptr; }
  if (conn){ delete [] conn; }
  if (continuous_data){ delete [] continuous_data; }
  if (element_data){ delete [] element_data; }
  if (data_file){ data_file->decref(); }
}

int TACSFH5Loader::loadData( const char *conn_fname,
                             const char *data_fname ){
  // Load in the data for the connectivity
  TACSFH5File *conn_file = new TACSFH5File(MPI_COMM_SELF);
  conn_file->incref();

  // Open the connectivity file
  int fail = conn_file->openFile(conn_fname);
  if (fail){
    conn_file->decref();
    conn_file = NULL;
    return fail;
  }

  conn_file->firstZone();
  int iterate = 1;
  while (iterate){
    const char *zone_name, *var_names;
    TACSFH5File::FH5DataType dtype;
    int dim1, dim2;

    // Get the name of the zone and its dimensions
    conn_file->getZoneInfo(&zone_name, &var_names, &dtype, &dim1, &dim2);

    if (strcmp("components", zone_name) == 0){
      void *idata;
      conn_file->getZoneData(NULL, NULL, NULL, NULL, NULL, &idata);
      comp_nums = (int*)idata;
      if (num_elements < 0){
        num_elements = dim1;
      }
      else if (num_elements != dim1){
        fprintf(stderr, "TACSFH5Loader: Number of elements is inconsistent\n");
        return 1;
      }
    }
    else if (strcmp("ltypes", zone_name) == 0){
      void *idata;
      conn_file->getZoneData(NULL, NULL, NULL, NULL, NULL, &idata);
      ltypes = (int*)idata;
      if (num_elements < 0){
        num_elements = dim1;
      }
      else if (num_elements != dim1){
        fprintf(stderr, "TACSFH5Loader: Number of elements is inconsistent\n");
        return 1;
      }
    }
    else if (strcmp("ptr", zone_name) == 0){
      void *idata;
      conn_file->getZoneData(NULL, NULL, NULL, NULL, NULL, &idata);
      ptr = (int*)idata;
      if (num_elements < 0){
        num_elements = dim1-1;
      }
      else if (num_elements != dim1-1){
        fprintf(stderr, "TACSFH5Loader: Number of elements is inconsistent\n");
        return 1;
      }
    }
    else if (strcmp("connectivity", zone_name) == 0){
      void *idata;
      conn_file->getZoneData(NULL, NULL, NULL, NULL, NULL, &idata);
      conn = (int*)idata;
      conn_size = dim1;
    }

    if (!conn_file->nextZone()){
      iterate = 0;
    }
  }

  conn_file->close();
  conn_file->decref();

  if (num_elements >= 0 && conn_size >= 0){
    if (!data_fname){
      data_fname = conn_fname;
    }

    // Load in the data for the connectivity
    data_file = new TACSFH5File(MPI_COMM_SELF);
    data_file->incref();

    // Open the connectivity file
    fail = data_file->openFile(data_fname);
    if (fail){
      data_file->decref();
      data_file = NULL;
      return fail;
    }

    iterate = 1;
    while (iterate){
      const char *zone_name, *var_names;
      TACSFH5File::FH5DataType dtype;
      int dim1, dim2;

      // Get the name of the zone and its dimensions
      data_file->getZoneInfo(&zone_name, &var_names, &dtype, &dim1, &dim2);

      if (strncmp("continuous data", zone_name, 15) == 0){
        void *fdata;
        data_file->getZoneData(&continuous_zone, &continuous_vars,
                               NULL, NULL, NULL, &fdata);
        num_nodes_continuous = dim1;
        num_vals_continuous = dim2;
        continuous_data = (float*)fdata;
      }
      else if (strncmp("element data", zone_name, 12) == 0){
        void *fdata;
        data_file->getZoneData(&element_zone, &element_vars,
                               NULL, NULL, NULL, &fdata);
        num_nodes_element = dim1;
        num_vals_element = dim2;
        element_data = (float*)fdata;
      }

      if (!data_file->nextZone()){
        iterate = 0;
      }
    }
  }

  return 0;
}

/**
   Get the number of components defined in this file
*/
int TACSFH5Loader::getNumComponents(){
  if (data_file){
    return data_file->getNumComponents();
  }
  return 0;
}

/**
   Return the component name

   @param comp The component number
   @return The component name
*/
char *TACSFH5Loader::getComponentName( int comp ){
  if (data_file){
    return data_file->getComponentName(comp);
  }
  return NULL;
}

/*
  Get the connectivity data from the file

  @param _num_elements The number of elements
  @param _comp_nums The component types

*/
void TACSFH5Loader::getConnectivity( int *_num_elements,
                                     int **_comp_nums, int **_ltypes,
                                     int **_ptr, int **_conn ){
  if (_num_elements){ *_num_elements = num_elements; }
  if (_comp_nums){ *_comp_nums = comp_nums; }
  if (_ltypes){ *_ltypes = ltypes; }
  if (_ptr){ *_ptr = ptr; }
  if (_conn){ *_conn = conn; }
}

void TACSFH5Loader::getContinuousData( const char **zone_name,
                                       const char **var_names,
                                       int *dim1, int *dim2, float **data ){
  if (zone_name){ *zone_name = continuous_zone; }
  if (var_names){ *var_names = continuous_vars; }
  if (dim1){ *dim1 = num_nodes_continuous; }
  if (dim2){ *dim2 = num_vals_continuous; }
  if (data){ *data = continuous_data; }
}

void TACSFH5Loader::getElementData( const char **zone_name,
                                    const char **var_names,
                                    int *dim1, int *dim2, float **data ){
  if (zone_name){ *zone_name = element_zone; }
  if (var_names){ *var_names = element_vars; }
  if (dim1){ *dim1 = num_nodes_element; }
  if (dim2){ *dim2 = num_vals_element; }
  if (data){ *data = element_data; }
}

/**
  Create a continuous representation of the element-wise data by
  averaging the element data to the nodes.

  @param index Index of the element data to be ordered
  @param data Output array of length the number of continuous nodes
*/
void TACSFH5Loader::getElementDataAsContinuous( int index,
                                                float *data ){
  if (conn && ptr && element_data){
    memset(data, 0, num_nodes_continuous*sizeof(float));
    if (index >= 0 && index < num_vals_element){
      int *count = new int[ num_nodes_continuous ];
      memset(count, 0, num_nodes_continuous*sizeof(int));
      for ( int i = 0; i < num_elements; i++ ){
        for ( int j = ptr[i]; j < ptr[i+1]; j++ ){
          data[conn[j]] += element_data[j*num_vals_element + index];
          count[conn[j]] += 1;
        }
      }

      for ( int i = 0; i < num_nodes_continuous; i++ ){
        data[i] /= count[i];
      }
      delete [] count;
    }
  }
}

/**
  Compute a value mask.

  This masks the element, places a non-zero entry in an array
  of length the number of elements, if the average element value of the
  specified index is between lower and upper bounds.

  @param layout Mask is computed only for the specified layout type
  @param use_continuous_data Truth flag for continuous or element data
  @param index Index of the data in the continuous/element data array
  @param lower Lower bound
  @param upper Upper bound
  @param mask Mask array the length of the number of elements
*/
void TACSFH5Loader::computeValueMask( ElementLayout layout,
                                      int use_continuous_data,
                                      int index,
                                      float lower, float upper,
                                      int *mask ){
  if (mask && conn && ptr && ltypes){
    if (use_continuous_data && continuous_data &&
        index >= 0 && index < num_vals_continuous){
      for ( int i = 0; i < num_elements; i++ ){
        // Do not mask the element by default.
        mask[i] = 0;

        if (ltypes[i] == layout){
          // Get the average value of the quantity
          if (ptr[i+1] - ptr[i] > 0){
            float val = 0.0;
            for ( int j = ptr[i]; j < ptr[i+1]; j++ ){
              val += continuous_data[conn[j]*num_vals_continuous + index];
            }
            val /= (ptr[i+1] - ptr[i]);

            if (val >= lower && val < upper){
              mask[i] = 1;
            }
          }
        }
      }
    }
    else if (!use_continuous_data && element_data &&
             index >= 0 && index < num_vals_element){
      for ( int i = 0; i < num_elements; i++ ){
        // Do not mask the element by default.
        mask[i] = 0;

        if (ltypes[i] == layout){
          // Get the average value of the quantity
          if (ptr[i+1] - ptr[i] > 0){
            float val = 0.0;
            for ( int j = ptr[i]; j < ptr[i+1]; j++ ){
              val += element_data[j*num_vals_continuous + index];
            }
            val /= (ptr[i+1] - ptr[i]);

            if (val >= lower && val < upper){
              mask[i] = 1;
            }
          }
        }
      }
    }
  }
}

/**
  Compute a mask array based on whether the average position of the element
  lies on the positive or negative side of a plane that passed through
  the specified base point. Points on the positive side are masked.

  @param layout Mask only elements which match the specified layout
  @param base Base point for the plane
  @param normal Normal direction for the plane
  @param mask Output array containing the specified mask values
*/
void TACSFH5Loader::computePlanarMask( ElementLayout layout,
                                       const float base[],
                                       const float normal[],
                                       int *mask ){
  if (mask && conn && ptr && ltypes){
    if (continuous_data && num_vals_continuous >= 3){
      for ( int i = 0; i < num_elements; i++ ){
        // Do not mask the element by default.
        mask[i] = 0;

        if (ltypes[i] == layout){
          // Get the average value of the position
          if (ptr[i+1] - ptr[i] > 0){
            float x[3] = {0.0, 0.0, 0.0};
            for ( int j = ptr[i]; j < ptr[i+1]; j++ ){
              x[0] += continuous_data[conn[j]*num_vals_continuous];
              x[1] += continuous_data[conn[j]*num_vals_continuous + 1];
              x[2] += continuous_data[conn[j]*num_vals_continuous + 2];
            }
            double inv = 1.0/(ptr[i+1] - ptr[i]);
            x[0] *= inv;
            x[1] *= inv;
            x[2] *= inv;

            // Check which side of the plane we are on. This
            // expression will suffer from subtractive cancellation
            // but here, since we're just concerned with visualization
            // we do the simple check. Geometric predicate could fix this..
            double dot = ((x[0] - base[0])*normal[0] +
                          (x[1] - base[1])*normal[1] +
                          (x[2] - base[2])*normal[2]);

            if (dot >= 0.0){
              mask[i] = 1;
            }
          }
        }
      }
    }
  }
}

/*
  Compute the unique set of triangles representing the
  faces
*/
/*
void TACSFH5Loader::getUnmatchedEdgesAndFaces( ElementLayout layout,
                                               const int *mask,
                                               int *_num_edges,
                                               int **_edges,
                                               int *_num_faces,
                                               int **_faces ){
  int num_edges = 0;
  int *edges = NULL;

  if (layout == TACS_HEXA_ELEMENT ||
      layout == TACS_HEXA_QUADRATIC_ELEMENT ||
      layout == TACS_HEXA_CUBIC_ELEMENT ||
      layout == TACS_HEXA_QUARTIC_ELEMENT ||
      layout == TACS_HEXA_QUARTIC_ELEMENT){

    int npe = 2;
    if (layout == TACS_HEXA_QUADRATIC_ELEMENT){
      npe = 3;
    }
    else if (layout == TACS_HEXA_CUBIC_ELEMENT){
      npe = 4;
    }
    else if (layout == TACS_HEXA_QUARTIC_ELEMENT){
      npe = 5;
    }
    else if (layout == TACS_HEXA_QUARTIC_ELEMENT){
      npe = 6;
    }

    // Scan through the elements, and figure out how many times
    // each node is referred to
    int *node_to_element_ptr = new int[ num_nodes_continuous+1 ];
    memset(node_to_element_ptr, 0, (num_nodes_continuous + 1)*sizeof(int));
    for ( int i = 0; i < num_elements; i++ ){
      if (ltypes[i] == layout && !(mask && mask[i]))){
        // Scan through each node
        for ( int j = 0; j < 8; j++ ){
          // Only consider the corner nodes
          int index = ((npe-1)*(j % 2) +
                       (npe-1)*npe*((j % 4)/2) +
                       (npe-1)*npe*npe*(j / 4));

          int node = conn[ptr[i] + index];
          node_to_element_ptr[node + 1]++;
        }
      }
    }

    // Set the node pointer
    for ( int i = 1; i <= num_nodes_continuous; i++ ){
      node_to_element_ptr[i] += node_to_element_ptr[i-1];
    }

    // Now, loop over and set the adjacent
    int *node_to_element = new int[ node_element_ptr[num_nodes_continuous] ];

    for ( int i = 0; i < num_elements; i++ ){
      if (ltypes[i] == layout && !(mask && mask[i]))){
        // Scan through each node
        for ( int j = 0; j < 8; j++ ){
          // Only consider the corner nodes
          int index = ((npe-1)*(j % 2) +
                       (npe-1)*npe*((j % 4)/2) +
                       (npe-1)*npe*npe*(j / 4));

          int node = conn[ptr[i] + index];
          node_to_element[node_to_element_ptr[node]] = i;
          node_to_element_ptr[node]++;
        }
      }
    }

    // Reset the node to element pointer
    for ( int i = num_nodes_continuous; i > 0; i-- ){
      node_to_element_ptr[i] = node_to_element_ptr[i-1];
    }
    node_to_element_ptr[0] = 0;

    // Look for elements that are adjacent to one another
    // via a connected face
    for ( int i = 0; i < num_elements; i++ ){
      for ( int face = 0; face < 6; face++ ){
        // Check for a common element index that is not
        // this element
        if (checkForCommonElement()){


          // Add the edges - there may be duplicates so
          for ( int k = 0; k < 4; k++ ){
            // Look for the node indices associated with the edge
            // for this face
            int j1 = hex_face_to_edges[face][k][0];
            int j2 = hex_face_to_edges[face][k][1];

            // Look for the closest two adjacent elements
            int n1 = ((npe-1)*(j1 % 2) +
                      (npe-1)*npe*((j1 % 4)/2) +
                      (npe-1)*npe*npe*(j1 / 4));
            int n2 = ((npe-1)*(j2 % 2) +
                      (npe-1)*npe*((j2 % 4)/2) +
                      (npe-1)*npe*npe*(j2 / 4));

            n1 = conn[ptr[i + n1]];
            n2 = conn[ptr[i + n2]];

            // Find the edges
            int imin = n1;
            int imax = n2;
            if (n2 > n1){
              imin = n2;
              imax = n1;
            }

            hash->addEntry(imin, imax);
          }




        }
      }
    }
  }

  // Extract the final set of edges
  *_num_edges = num_edges;
  *_edges = edges;
}
*/

const int hex_ordering_transform[] = {0, 1, 3, 2, 4, 5, 7, 6};

/*
  Extract the iso-surface for the given element type.

  @param layout Element layout type to look for
  @param mask Element mask array to apply
  @param isoval The isovalue of the contour
  @param index Index of the continuous data set to use
  @param _data Continuuos data set to use (overrides index choice)
  @param _ntris Output number of triangles
  @param _verts Triangles vertices
*/
void TACSFH5Loader::getIsoSurfaces( ElementLayout layout,
                                    const int *mask,
                                    float isoval,
                                    int index,
                                    float *_data,
                                    int *_ntris,
                                    float **_verts ){
  // Set the proper pointer to the data
  int incr = 1;
  float *data = NULL;
  if (_data){
    data = _data;
  }
  else if (index >= 0 && index < num_vals_continuous){
    data = &continuous_data[index];
    incr = num_vals_continuous;
  }

  // Set the number of triangles/vertices
  int ntris = 0;
  float *verts = NULL;

  // Keep track of the max number of triangles
  int chunk_size = num_elements;
  if (num_elements > (1 << 16)){
    chunk_size = 1 << 16;
  }
  int max_num_tris = 0;

  if (layout == TACS_HEXA_ELEMENT ||
      layout == TACS_HEXA_QUADRATIC_ELEMENT ||
      layout == TACS_HEXA_CUBIC_ELEMENT ||
      layout == TACS_HEXA_QUARTIC_ELEMENT ||
      layout == TACS_HEXA_QUARTIC_ELEMENT){

    int npe = 2;
    if (layout == TACS_HEXA_QUADRATIC_ELEMENT){
      npe = 3;
    }
    else if (layout == TACS_HEXA_CUBIC_ELEMENT){
      npe = 4;
    }
    else if (layout == TACS_HEXA_QUARTIC_ELEMENT){
      npe = 5;
    }
    else if (layout == TACS_HEXA_QUARTIC_ELEMENT){
      npe = 6;
    }

    for ( int i = 0; i < num_elements; i++ ){
      if (ltypes[i] == layout && !(mask && mask[i])){
        for ( int iz = 0; iz < npe-1; iz++ ){
          for ( int iy = 0; iy < npe-1; iy++ ){
            for ( int ix = 0; ix < npe-1; ix++ ){
              TACSMarchingCubesCell cell;

              for ( int kk = 0; kk < 2; kk++ ){
                for ( int jj = 0; jj < 2; jj++ ){
                  for ( int ii = 0; ii < 2; ii++ ){
                    // Compute the index
                    int index = hex_ordering_transform[ii + 2*jj + 4*kk];

                    // Compute the offset into the local mesh
                    int offset =
                      (ix + ii) +
                      (iy + jj)*npe +
                      (iz + kk)*npe*npe;

                    // Get the node number
                    int node = conn[ptr[i] + offset];
                    cell.val[index] = data[incr*node];

                    // Extract the node value
                    const float *vals = &continuous_data[node*num_vals_continuous];
                    cell.p[index].x = vals[0];
                    cell.p[index].y = vals[1];
                    cell.p[index].z = vals[2];
                  }
                }
              }

              // Find the additional triangles that are needed
              TACSMarchingCubesTriangle tris[5];
              int new_tris = TacsPolygonizeCube(cell, isoval, tris);

              // Expand the buffer if needed
              if (new_tris + ntris > max_num_tris){
                max_num_tris = ntris + new_tris + chunk_size;
                float *buff = new float[ 9*max_num_tris ];
                memcpy(buff, verts, 9*ntris*sizeof(float));
                delete [] verts;
                verts = buff;
              }

              // Add the new triangles
              float *v = &verts[9*ntris];
              for ( int k = 0; k < new_tris; ntris++, k++ ){
                for ( int kk = 0; kk < 3; kk++ ){
                  v[0] = tris[k].p[kk].x;
                  v[1] = tris[k].p[kk].y;
                  v[2] = tris[k].p[kk].z;
                  v += 3;
                }
              }
            }
          }
        }
      }
    }
  }

  *_ntris = ntris;
  *_verts = verts;
}
