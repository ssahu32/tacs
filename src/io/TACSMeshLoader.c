#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tacsmetis.h"

#include "FElibrary.h"
#include "TACSMeshLoader.h"
#include "FSDTStiffness.h"
#include "TensorToolbox.h"
/* #include "EBStiffness.h" */
/* #include "EBBeam.h" */

/*!
  This is an interface for reading NASTRAN-style files.

  Copyright (c) 2011 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.

  Description of the Nastran input file:

  It is a fixed-width format. There are five sections in a Nastran
  file - often called an input deck.

  1. Nastran statement
  2. File management section
  3. Executive control section
  CEND
  4. Case control section - load cases defined
  BEGIN BULK
  5. Bulk data section
  ENDDATA

  The first two sections are optional. The delimiters are not
  optional. 

  Format of bulk data: 
  - Each line must have 80 columns. 
  - A bulk data entry may span multiple lines
  - Three types of data formats: integer, real, character string
  - Each field in the file has a specific input format - the quick
  reference guide specifies the format for each type of element/data
  type.
  
  - There are three types of field formats: small field, large field, 
  and free field format

  Small field format:
  - Each line is divided into 10 fields, 8 columns wide

  Large field format:
  - Each line is divided into 6 fields, the first is 8 columns wide, 
  the next 4 are 16 wide and the last is 8 columns wide. The large field
  format is signified by an asterisk after the keyword.
  
  Free field format:
  - Fields separated by commas, only precise to eight digits - higher
  precision truncated.

  Continuation entries: 

  Method 1: +M101, +M101 on the preceeding and continuation entries.
  Method 2: Last entry of preceeding and first entry of the continuation
  line left blank.

  - Input data in fields 1, 10 must be left justified
  - All real numbers must have a decimal place - including zero
  - Default values may be left blank
  - Comments start with a dollar sign  
*/

/*
  Extend an integer array 
*/
static void extend_int_array( int ** array, int old_len, 
			      int new_len ){
  int * temp = new int[ new_len ];
  memcpy(temp, *array, sizeof(int)*old_len);
  delete [] *array;
  *array = temp;
}

/*
  Functions for sorting a list such that:

  arg_sort_list[list[i]] is in ascending order
*/
static const int * arg_sort_list = NULL;

static int compare_arg_sort( const void * a, const void * b ){
  return arg_sort_list[*(int*)a] - arg_sort_list[*(int*)b];
}

/*
  Read a line from the buffer.

  Return the number of read characters. Do not exceed the buffer
  length len.

  Given the line buffer 'line', and the size of the line buffer line_len

*/
static int read_buffer_line( char * line, size_t line_len, 
                             size_t * loc, char * buffer, size_t buffer_len ){
  size_t i = 0;
  for ( ; (i < line_len) && (*loc < buffer_len); i++, (*loc)++ ){
    if (buffer[*loc] == '\n'){
      break;
    }
    line[i] = buffer[*loc];    
  }

  // Read until the end of the line or
  while ((*loc < buffer_len) && (buffer[*loc] != '\n')){
    (*loc)++;
  }

  (*loc)++; // Increment so that loc is at the next location
  
  return i;
}

/*
  Reverse look up.

  Given the sorted array list[arg[k]], find k, such that 

  list[arg[k]] = var
*/
static int find_index_arg_sorted( int var, int size, 
				  const int * list, const int * args ){
  // Binary search an array to find k such that list[k] = var,
  // where the array list[args[k]] is sorted in ascending
  // order
  int high = size-1;
  int low = 0;
  int high_val = list[args[high]];
  int low_val = list[args[low]];

  // Check if the index is at the end points
  if (var == low_val){
    return low;
  }
  else if (var < low_val){
    return -1;
  }

  if (var == high_val){
    return high;
  }
  else if (var > high_val){
    return -1;
  }

  int mid = low + (int)((high - low)/2);
      
  // While there are values left in the list
  while (low != mid){
    int mid_val = list[args[mid]];
    if (mid_val == var){
      return mid;
    }	
    
    if (var < mid_val){
      high = mid;
      high_val = mid_val;
    }
    else {
      low = mid;
      low_val = mid_val;
    }
    
    mid = low + (int)((high - low)/2);
  }      
  
  return -1;
}

/*
  Convert a Nastran-style number with an exponent to a double.
*/
static double bdf_atof( char * str ){
  // First, check if the string contains an E/e or D/d - if so, convert it
  int slen = strlen(str);
  for ( int i = 0; i < slen; i++ ){
    if (str[i] == 'e' || str[i] == 'E'){
      return atof(str);
    }
    else if (str[i] == 'd' || str[i] == 'D'){
      str[i] = 'e';
      return atof(str);
    }
  }

  // Convert the special Nastran number format without e/E or d/D 
  // 4.-4 or 5.34+2 etc.
  char temp[24];
  int i = 0, j = 0;
  while (i < slen && str[i] == ' '){ i++; }
  if (i == slen){ 
    return 0.0;
  }

  // Take care of a leading minus sign
  if (str[i] == '-' ){ 
    temp[j] = str[i];
    j++, i++; 
  }

  for ( ; i < slen; i++, j++ ){
    // Add the character 'e' before the exponent
    if (str[i] == '-' || str[i] == '+'){
      temp[j] = 'e'; j++; 
    }
    temp[j] = str[i];
  }
  temp[j] = '\0';

  return atof(temp);
}

/*
  Parse the long-field format
  
  The long field format is split into the fixed widths:

  0 --- 8 ---24--- 40--- 56--- 72 ---80

  GRID* num  coord x     y      
        z

  This code ignores the coordinate info 
*/
static void parse_node_long_field( char * line, char * line2, int * node, 
                                   double * x, double * y, double * z ){
  char Node[32], X[32], Y[32], Z[32];

  strncpy(Node, &line[8], 16); Node[16] = '\0';
  strncpy(X, &line[40], 16);   X[16] = '\0';
  strncpy(Y, &line[56], 16);   Y[16] = '\0';
  strncpy(Z, &line2[8], 16);   Z[16] = '\0';

  *node = atoi(Node);
  *x = bdf_atof(X);
  *y = bdf_atof(Y);
  *z = bdf_atof(Z);
}

/*
  Parse the short-field or free-field comma separated format

  The short field format is fixed-width as follows:

  0 --- 8 --- 16 --- 24 --- 32 --- 40
  GRID  num   coord  x      y      z 
*/
static void parse_node_short_free_field( char * line, int * node, 
                                         double * x, double * y, double * z ){
  char field[5][32];

  // Look for a comma 
  int len = strlen(line);
  int comma_format = 0;
  for ( int i = 0; (i < len) && (i < 80); i++ ){
    if (line[i] == ','){ 
      comma_format = 1;
      break;
    }
  }

  if (comma_format){
    int start = 8;
    int end = 8;
    for ( int i = 0; i < 5; i++ ){
      start = end;
      while (end < len && line[end] != ',')
	end++;
      int flen = end-start;
      strncpy(field[i], &line[start], flen);
      field[i][flen] = '\0';
      end++;
    }
  }
  else { // Short-format, fixed width
    strncpy(field[0], &line[8], 8);   field[0][8] = '\0';
    strncpy(field[2], &line[24], 8);  field[2][8] = '\0';
    strncpy(field[3], &line[32], 8);  field[3][8] = '\0';
    strncpy(field[4], &line[40], 8);  field[4][8] = '\0';
  }

  *node = atoi(field[0]);
  *x = bdf_atof(field[2]);
  *y = bdf_atof(field[3]);
  *z = bdf_atof(field[4]);
}

/*
  Parse an element
*/
static void parse_element_field( char line[], 
				 int * elem_num, int * component_num,
				 int * node_nums, int num_nodes ){
  int n = 0; // The number of parsed nodes
  char node[9];
  int entry = 8;

  if (n == 0){ 
    strncpy(node, &line[entry], 8);
    node[8] = '\0';
    *elem_num = atoi(node);
    entry += 8;
    
    strncpy(node, &line[entry], 8);
    node[8] = '\0';
    *component_num = atoi(node);
    entry += 8;
  }

  if (*component_num <= 0){
    fprintf(stderr, 
	    "Error: The component numbers must be strictly positive\n");
  }
  
  for ( ; n < num_nodes && entry < 80; entry += 8, n++ ){
    // Parse the line containing the entry
    strncpy(node, &line[entry], 8);
    node[8] = '\0';
    node_nums[n] = atoi(node);
  }
}

static void parse_element_field2( char line1[], char line2[],
                                  int * elem_num, int * component_num,
                                  int * node_nums, int num_nodes ){

  int n = 0; // The number of parsed nodes
  char node[9];

  for ( int m = 0; m < 2; m++ ){
    int entry = 8;
    const char * line = line1;
    if (m == 1){
      line = line2;
    }

    if (n == 0){ 
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *elem_num = atoi(node);
      entry += 8;

      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *component_num = atoi(node);
      entry += 8;
    }

    for ( ; n < num_nodes && entry < 72; entry += 8, n++ ){
      // Parse the line containing the entry
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      node_nums[n] = atoi(node);
    }
  }
}

static void parse_element_field3( char line1[], char line2[], char line3[],
                                  int * elem_num, int * component_num,
                                  int * node_nums, int num_nodes ){  
  int n = 0; // The number of parsed nodes
  char node[9];

  for ( int m = 0; m < 3; m++ ){
    int entry = 8;
    const char * line = line1;
    if (m == 1){
      line = line2;
    }
    else if (m == 2){
      line = line3;
    }

    if (n == 0){ 
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *elem_num = atoi(node);
      entry += 8;

      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *component_num = atoi(node);
      entry += 8;
    }

    for ( ; n < num_nodes && entry < 72; entry += 8, n++ ){
      // Parse the line containing the entry
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      node_nums[n] = atoi(node);
    }
  }
}

static void parse_element_field4( char line1[], char line2[], char line3[],
                                  char line4[],
                                  int * elem_num, int * component_num,
                                  int * node_nums, int num_nodes ){  
  int n = 0; // The number of parsed nodes
  char node[9];

  for ( int m = 0; m < 4; m++ ){
    int entry = 8;
    const char * line = line1;
    if (m == 1){
      line = line2;
    }
    else if (m == 2){
      line = line3;
    }
    else if (m == 3){
      line = line4;
    }

    if (n == 0){ 
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *elem_num = atoi(node);
      entry += 8;

      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *component_num = atoi(node);
      entry += 8;
    }

    for ( ; n < num_nodes && entry < 72; entry += 8, n++ ){
      // Parse the line containing the entry
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      node_nums[n] = atoi(node);
    }
  }
}

static void parse_element_field9( char line1[], char line2[], char line3[],
                                  char line4[], char line5[], char line6[],
                                  char line7[], char line8[], char line9[],
                                  int * elem_num, int * component_num,
                                  int * node_nums, int num_nodes ){  
  int n = 0; // The number of parsed nodes
  char node[9];

  for ( int m = 0; m < 4; m++ ){
    int entry = 8;
    const char * line = line1;
    if (m == 1){ line = line2; }
    else if (m == 2){ line = line3; }
    else if (m == 3){ line = line4; }
    else if (m == 4){ line = line5; }
    else if (m == 5){ line = line6; }
    else if (m == 6){ line = line7; }
    else if (m == 7){ line = line8; }
    else if (m == 8){ line = line9; }
    
    if (n == 0){ 
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *elem_num = atoi(node);
      entry += 8;

      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      *component_num = atoi(node);
      entry += 8;
    }

    for ( ; n < num_nodes && entry < 72; entry += 8, n++ ){
      // Parse the line containing the entry
      strncpy(node, &line[entry], 8);
      node[8] = '\0';
      node_nums[n] = atoi(node);
    }
  }
}

TACSMeshLoader::TACSMeshLoader( MPI_Comm _comm ){
  comm = _comm;

  // Initialize everything to zero
  num_nodes = num_elements = 0;
  num_bcs = 0;
  node_nums = NULL;
  Xpts_unsorted = NULL;
  elem_node_con = elem_node_ptr = NULL;
  elem_component = NULL;
  Xpts = NULL;
  bc_vals = NULL;
  bc_nodes = bc_con = bc_ptr = NULL;
  orig_bc_vals = NULL;
  orig_bc_nodes = orig_bc_con = orig_bc_ptr = NULL;
  num_components = 0;
  elements = NULL;
  component_elems = NULL;
  component_descript = NULL;
  num_owned_elements = 0; 
  local_component_nums = NULL;
}

TACSMeshLoader::~TACSMeshLoader(){
  if (elem_node_con){ delete [] elem_node_con; }
  if (elem_node_ptr){ delete [] elem_node_ptr; }
  if (elem_component){ delete [] elem_component; }
  if (Xpts){ delete [] Xpts; }
  if (bc_nodes){ delete [] bc_nodes; }
  if (bc_con){ delete [] bc_con; }
  if (bc_vals){ delete [] bc_vals; }
  if (bc_ptr){ delete [] bc_ptr; }

  if (orig_bc_nodes){ delete [] orig_bc_nodes; }
  if (orig_bc_con){ delete [] orig_bc_con; }
  if (orig_bc_vals){ delete [] orig_bc_vals; }
  if (orig_bc_ptr){ delete [] orig_bc_ptr; }

  if (elements){
    for ( int k = 0; k < num_components; k++ ){
      if (elements[k]){ elements[k]->decref(); }
    }
    delete [] elements;
  }
  if (component_elems){ delete [] component_elems; }
  if (component_descript){ delete [] component_descript; }
  if (local_component_nums){ delete [] local_component_nums; }
  if (Xpts_unsorted){ delete [] Xpts_unsorted;}
  if (node_nums) {delete [] node_nums;}
}

/*
  Get the number of components defined by the data
*/
int TACSMeshLoader::getNumComponents(){
  return num_components;
}

/*
  Set the element associated with a given component number
*/
void TACSMeshLoader::setElement( int component_num, 
				 TACSElement * _element ){
  if (_element && (component_num >= 0) && (component_num < num_components)){
    _element->incref();
    elements[component_num] = _element;
  }
}

const char * TACSMeshLoader::getComponentDescript( int comp_num ){
  if (component_descript && (comp_num >= 0) && 
      (comp_num < num_components)){
    return &component_descript[33*comp_num];
  }
  return NULL;
}

/*
  Retrieve the element description corresponding to the component number
*/
const char * TACSMeshLoader::getElementDescript( int comp_num ){
  if (component_elems && (comp_num >= 0) && 
      (comp_num < num_components)){
    return &component_elems[9*comp_num];
  }
  return NULL; // No associated element
}

/*
  This scans a Nastran file - only scanning in information from the
  bulk data section.

  The only entries scanned are the entries beginning with elem_types
  and any GRID/GRID* entries.
*/
int TACSMeshLoader::scanBdfFile( const char * file_name ){
  int rank;
  MPI_Comm_rank(comm, &rank);
  int fail = 0;
  int root = 0;

  if (rank == root){
    FILE * fp = fopen(file_name, "r"); 
    if (!fp){ 
      fprintf(stderr, "TACSMeshLoader: Unable to open file %s\n", file_name);
      fail = 1;
      MPI_Abort(comm, fail);
      return fail;
    }
    
    // Count up the number of nodes, elements and size of connectivity data
    num_nodes = 0;
    num_elements = 0;
    num_components = 0;
    num_bcs = 0;

    // The size of the connectivity arrays
    int elem_con_size = 0;
    int bc_con_size = 0;

    // Each line can only be 80 characters long
    char line[9][80];

    // Determine the size of the file
    fseek(fp, 0, SEEK_END);
    size_t buffer_len = ftell(fp);
    rewind(fp);

    // Allocate enough space to store the entire file
    char * buffer = new char[buffer_len];
    if (fread(buffer, 1, buffer_len, fp) != buffer_len){
      fprintf(stderr, "[%d] TACSMeshLoader: Problem reading file %s\n",
              rank, file_name);
      MPI_Abort(comm, 1);
      return 1;
    }
    fclose(fp);

    // Keep track of where the current point in the buffer is
    size_t buffer_loc = 0;
    read_buffer_line(line[0], sizeof(line[0]), 
                     &buffer_loc, buffer, buffer_len);

    int node;
    double x, y, z;
    int inBulk = 0;
    int bulk_start = 0;
    while (buffer_loc < buffer_len){
      // We don't start recording anything until we have detected the
      // BEGIN BULK command.
      if (inBulk == 0){
	if (strncmp(line[0], "BEGIN BULK", 10) == 0){
	  inBulk = 1;
	  bulk_start = buffer_loc;
	}
	read_buffer_line(line[0], sizeof(line[0]), 
			 &buffer_loc, buffer, buffer_len);
      }
      else {	
	if (line[0][0] != '$'){ // A comment line
	  // Check for GRID or GRID*
	  if (strncmp(line[0], "GRID*", 5) == 0){
	    if (!read_buffer_line(line[1], sizeof(line[1]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1;
	      break;
	    }
	    parse_node_long_field(line[0], line[1], &node, &x, &y, &z);
	    num_nodes++;
	  }
	  else if (strncmp(line[0], "GRID", 4) == 0){
	    parse_node_short_free_field(line[0], &node, &x, &y, &z);
	    num_nodes++;
	  }
	  else if (strncmp(line[0], "CBAR", 4) == 0){
	    // Read in the component number and nodes associated with
	    // this element
	    int elem_num, component_num;
	    int nodes[2]; // Should have at most four nodes
	    parse_element_field(line[0],
				&elem_num, &component_num,
				nodes, 2);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 2;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CHEXA", 5) == 0){
	    if (!read_buffer_line(line[1], sizeof(line[1]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1; break;
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num, nodes[8]; 
	    parse_element_field2(line[0], line[1], 
				 &elem_num, &component_num, 
				 nodes, 8);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 8;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CHEXA27", 7) == 0){
	    for ( int i = 1; i < 4; i++ ){
	      if (!read_buffer_line(line[i], sizeof(line[i]), 
				    &buffer_loc, buffer, buffer_len)){
		fail = 1; break;
	      }
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num, nodes[27];
	    parse_element_field4(line[0], line[1], line[2], line[3],
				 &elem_num, &component_num, 
				 nodes, 27);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 27;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CHEXA64", 7) == 0){
	    for ( int i = 1; i < 9; i++ ){
	      if (!read_buffer_line(line[i], sizeof(line[i]), 
				    &buffer_loc, buffer, buffer_len)){
		fail = 1; break;
	      }
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num, nodes[27];
	    parse_element_field9(line[0], line[1], line[2], line[3], line[4],
				 line[5], line[6], line[7], line[8],
				 &elem_num, &component_num, 
				 nodes, 64);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 64;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CQUAD16", 7) == 0){
	    if (!read_buffer_line(line[1], sizeof(line[1]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1; break;
	    }
	    if (!read_buffer_line(line[2], sizeof(line[2]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1; break;
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num;
	    int nodes[16]; // Should have at most four nodes
	    parse_element_field3(line[0], line[1], line[2],
				 &elem_num, &component_num,
				 nodes, 16);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 16;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CQUAD9", 6) == 0){
	    if (!read_buffer_line(line[1], sizeof(line[1]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1; break;
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num;
	    int nodes[9]; // Should have at most four nodes
	    parse_element_field2(line[0], line[1],
				 &elem_num, &component_num,
				 nodes, 9);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 9;
	    num_elements++;
	  }
	  else if (strncmp(line[0], "CQUAD4", 6) == 0 || 
		   strncmp(line[0], "CQUADR", 6) == 0){
	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num;
	    int nodes[4]; // Should have at most four nodes
	    parse_element_field(line[0],
				&elem_num, &component_num,
				nodes, 4);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 4;
	    num_elements++;
	  }  
	  else if (strncmp(line[0], "CQUAD", 5) == 0){
	    if (!read_buffer_line(line[1], sizeof(line[1]), 
				  &buffer_loc, buffer, buffer_len)){
	      fail = 1; break;
	    }

	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num;
	    int nodes[9]; // Should have at most four nodes
	    parse_element_field2(line[0], line[1],
				 &elem_num, &component_num,
				 nodes, 9);

	    if (component_num > num_components){
	      num_components = component_num;
	    }

	    elem_con_size += 9;
	    num_elements++;
	  }      
	  else if (strncmp(line[0], "SPC", 3) == 0){
	    bc_con_size += 6;
	    num_bcs++;
	  }
	  else if (strncmp(line[0], "FFORCE", 6) == 0){	  
	    // Read in the component number and nodes associated with this element
	    int elem_num, component_num;
	    int nodes[1]; // Should have at most four nodes
	    parse_element_field(line[0],
				&elem_num, &component_num,
				nodes, 1);

	    if (component_num > num_components){
	      num_components = component_num;
	    }
	    elem_con_size += 1;
	    num_elements++;
	  }
	}

	read_buffer_line(line[0], sizeof(line[0]), 
			 &buffer_loc, buffer, buffer_len);
      }
    }

    // Allocate space for everything    
    node_nums = new int[ num_nodes ];
    Xpts_unsorted = new double[ 3*num_nodes ];
    
    // Element type information
    int * elem_nums = new int[ num_elements ];
    int * elem_comp = new int[ num_elements ];
    
    // The connectivity information
    int * elem_con = new int[ elem_con_size ];
    int * elem_con_ptr = new int[ num_elements+1 ];
    elem_con_ptr[0] = 0;
    
    // Boundary condition information
    bc_nodes = new int[ num_bcs ];
    bc_ptr = new int[ num_bcs+1 ];
    bc_vals = new double[ bc_con_size ];
    bc_con = new int[ bc_con_size ];
    bc_ptr[0] = 0;

    // Allocate space for storing the component names
    component_elems = new char[ 9*num_components ];
    component_descript = new char[ 33*num_components ];
    memset(component_elems, '\0', 9*num_components*sizeof(char));
    memset(component_descript, '\0', 33*num_components*sizeof(char));

    // Reset the sizes of the things to be read in
    num_nodes = 0;
    num_elements = 0;
    num_bcs = 0;
    elem_con_size = 0;
    bc_con_size = 0;

    // Rewind to the beginning of the bulk section and allocate everything
    buffer_loc = bulk_start;
    read_buffer_line(line[0], sizeof(line[0]), 
                     &buffer_loc, buffer, buffer_len);

    // Keep track of the component numbers loaded from an
    // ICEM-generated bdf file
    int component_counter = 0;

    while (buffer_loc < buffer_len){
      if (strncmp(line[0], "$CDSCRPT", 8) == 0){
        // A non-standard - pyLayout specific - description of the
        // component. This is very useful for describing what the
        // components actually are with a string.
        // Again use a fixed width format
        char comp[33];
        strncpy(comp, &line[0][8], 16);
        comp[16] = '\0';
        int comp_num = atoi(comp)-1;
        strncpy(comp, &line[0][24], 32);
        comp[32] = '\0';
        // Remove white space
	if (comp_num >= 0 && comp_num < num_components){
	  sscanf(comp, "%s", &component_descript[33*comp_num]);
	}
      }
      else if (strncmp(line[0], "$       Shell", 13) == 0){
        // A standard icem output - description of each
        // component. This is very useful for describing what the
        // components actually are with a string.
        // Again use a fixed width format
        char comp[33];
        int comp_num = component_counter; 
	component_counter++;

        strncpy(comp, &line[0][41], 32);
        comp[32] = '\0';
        // Remove white space
	if (comp_num >= 0 && comp_num < num_components){
	  sscanf(comp, "%s", &component_descript[33*comp_num]);
	}
      }
      if (line[0][0] != '$'){ // A comment line
	// Check for GRID or GRID*
	if (strncmp(line[0], "GRID*", 5) == 0){
          if (!read_buffer_line(line[1], sizeof(line[1]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1;
            break;
          }
	  parse_node_long_field(line[0], line[1], &node, &x, &y, &z);
	  node_nums[num_nodes] = node-1; // Get the C ordering
	  Xpts_unsorted[3*num_nodes]   = x;
	  Xpts_unsorted[3*num_nodes+1] = y;
	  Xpts_unsorted[3*num_nodes+2] = z;
	  num_nodes++;
	}
	else if (strncmp(line[0], "GRID", 4) == 0){
	  parse_node_short_free_field(line[0], &node, &x, &y, &z);
	  node_nums[num_nodes] = node-1; // Get the C ordering
	  Xpts_unsorted[3*num_nodes]   = x;
	  Xpts_unsorted[3*num_nodes+1] = y;
	  Xpts_unsorted[3*num_nodes+2] = z;	
	  num_nodes++;
	}        
        else if (strncmp(line[0], "CBAR", 4) == 0){
          // Read in the component number and nodes associated with
          // this element
	  int elem_num, component_num;
	  int nodes[2]; // Should have at most two nodes
	  parse_element_field(line[0],
			      &elem_num, &component_num,
			      nodes, 2);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;

          elem_con[elem_con_size] = nodes[0]-1;
          elem_con[elem_con_size+1] = nodes[1]-1;

	  elem_con_size += 2;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CBAR");
          }
        }
        else if (strncmp(line[0], "CHEXA", 5) == 0){
          if (!read_buffer_line(line[1], sizeof(line[1]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1; break;
          }

          // Read in the component number and nodes associated with this element
	  int elem_num, component_num, nodes[8]; 
          parse_element_field2(line[0], line[1], 
                               &elem_num, &component_num, 
                               nodes, 8);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;

	  elem_con[elem_con_size]   = nodes[0]-1;
	  elem_con[elem_con_size+1] = nodes[1]-1;
	  elem_con[elem_con_size+2] = nodes[3]-1;
	  elem_con[elem_con_size+3] = nodes[2]-1;
	  elem_con[elem_con_size+4] = nodes[4]-1;
	  elem_con[elem_con_size+5] = nodes[5]-1;
	  elem_con[elem_con_size+6] = nodes[7]-1;
	  elem_con[elem_con_size+7] = nodes[6]-1;
         
	  elem_con_size += 8;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CHEXA");
          }
        }
        else if (strncmp(line[0], "CHEXA27", 7) == 0){
          for ( int i = 1; i < 4; i++ ){
            if (!read_buffer_line(line[i], sizeof(line[i]), 
                                  &buffer_loc, buffer, buffer_len)){
              fail = 1; break;
            }
          }

          // Read in the component number and nodes associated with this element
	  int elem_num, component_num, nodes[27];
          parse_element_field4(line[0], line[1], line[2], line[3],
                               &elem_num, &component_num, 
                               nodes, 27);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;
          
          for ( int k = 0; k < 27; k++ ){
            elem_con[elem_con_size+k] = nodes[k]-1;
          }

	  elem_con_size += 27;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CHEXA64");
          }
        }
        else if (strncmp(line[0], "CHEXA64", 7) == 0){
          for ( int i = 1; i < 9; i++ ){
            if (!read_buffer_line(line[i], sizeof(line[i]), 
                                  &buffer_loc, buffer, buffer_len)){
              fail = 1; break;
            }
          }

          // Read in the component number and nodes associated with this element
	  int elem_num, component_num, nodes[64];
          parse_element_field9(line[0], line[1], line[2], line[3], line[4],
                               line[5], line[6], line[7], line[8],
                               &elem_num, &component_num, 
                               nodes, 64);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;
          
          for ( int k = 0; k < 64; k++ ){
            elem_con[elem_con_size+k] = nodes[k]-1;
          }

	  elem_con_size += 64;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CHEXA64");
          }
        }
        else if (strncmp(line[0], "CQUAD16", 7) == 0){
          if (!read_buffer_line(line[1], sizeof(line[1]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1; break;
          }
          if (!read_buffer_line(line[2], sizeof(line[2]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1; break;
          }

	  // Read in the component number and nodes associated with this element
	  int elem_num, component_num;
	  int nodes[16]; // Should have at most four nodes
	  parse_element_field3(line[0], line[1], line[2],
                               &elem_num, &component_num,
                               nodes, 16);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;
          
          for ( int k = 0; k < 16; k++ ){
            elem_con[elem_con_size+k] = nodes[k]-1;
          }

	  elem_con_size += 16;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CQUAD16");
          }
        }
        else if (strncmp(line[0], "CQUAD9", 6) == 0){
          if (!read_buffer_line(line[1], sizeof(line[1]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1; break;
          }

	  // Read in the component number and nodes associated with this element
	  int elem_num, component_num;
	  int nodes[9]; // Should have at most four nodes
	  parse_element_field2(line[0], line[1],
                               &elem_num, &component_num,
                               nodes, 9);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;
          
          for ( int k = 0; k < 9; k++ ){
            elem_con[elem_con_size+k] = nodes[k]-1;
          }

	  elem_con_size += 9;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CQUAD9");
          }
        }
	else if (strncmp(line[0], "CQUAD4", 6) == 0 ||
		 strncmp(line[0], "CQUADR", 6) == 0){
	  // Read in the component number and nodes associated with this element
	  int elem_num, component_num;
	  int nodes[4]; // Should have at most four nodes
	  parse_element_field(line[0],
			      &elem_num, &component_num,
			      nodes, 4);
	  
	  // Add the element to the connectivity list
	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;

	  elem_con[elem_con_size]   = nodes[0]-1;
	  elem_con[elem_con_size+1] = nodes[1]-1;
	  elem_con[elem_con_size+2] = nodes[3]-1;
	  elem_con[elem_con_size+3] = nodes[2]-1;
	  elem_con_size += 4;

	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CQUAD4");
          }
	}
        else if (strncmp(line[0], "CQUAD", 5) == 0){
          if (!read_buffer_line(line[1], sizeof(line[1]), 
                                &buffer_loc, buffer, buffer_len)){
            fail = 1; break;
          }

	  // Read in the component number and nodes associated with this element
	  int elem_num, component_num;
	  int nodes[9]; // Should have at most four nodes
	  parse_element_field2(line[0], line[1],
                               &elem_num, &component_num,
                               nodes, 9);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;
          
	  elem_con[elem_con_size] = nodes[0]-1;
	  elem_con[elem_con_size+1] = nodes[4]-1;
	  elem_con[elem_con_size+2] = nodes[1]-1;

	  elem_con[elem_con_size+3] = nodes[7]-1;
	  elem_con[elem_con_size+4] = nodes[8]-1;
	  elem_con[elem_con_size+5] = nodes[5]-1;

	  elem_con[elem_con_size+6] = nodes[3]-1;
	  elem_con[elem_con_size+7] = nodes[6]-1;
	  elem_con[elem_con_size+8] = nodes[2]-1;          

	  elem_con_size += 9;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "CQUAD");
          }
        }
	else if (strncmp(line[0], "FFORCE", 6) == 0){
	  // Read in the component number and nodes associated 
	  // with the following force
	  int elem_num, component_num;
	  int nodes[1]; // Should have at most four nodes
	  parse_element_field(line[0], &elem_num, &component_num,
			      nodes, 1);

	  elem_nums[num_elements] = elem_num-1;
	  elem_comp[num_elements] = component_num-1;          
	  elem_con[elem_con_size] = nodes[0]-1;

	  elem_con_size++;
	  elem_con_ptr[num_elements+1] = elem_con_size;
	  num_elements++;

          if (component_elems[9*(component_num-1)] == '\0'){
            strcpy(&component_elems[9*(component_num-1)], "FFORCE");
          }
	}
	else if (strncmp(line[0], "SPC", 3) == 0){
	  // This is a variable-length format. Read in grid points until 
	  // zero is reached. This is a fixed-width format
	  // SPC SID  G1  C  D
	  
	  // Read in the nodal value
	  char node[9];
	  strncpy(node, &line[0][16], 8);
	  node[8] = '\0';
	  bc_nodes[num_bcs] = atoi(node)-1;
          
	  strncpy(node, &line[0][32], 8);
	  node[8] = '\0';
	  double val = bdf_atof(node);
	  
	  // Read in the dof that will be constrained
	  for ( int k = 24; k < 32; k++ ){
	    char dofs[7] = "123456";
	    
	    for ( int j = 0; j < 6; j++ ){
	      if (dofs[j] == line[0][k]){
		bc_con[bc_con_size] = j;
		bc_vals[bc_con_size] = val;
		bc_con_size++;
		break;
	      }
	    }
	  }
	  
	  bc_ptr[num_bcs+1] = bc_con_size;
	  num_bcs++;
	}
      }

      read_buffer_line(line[0], sizeof(line[0]), 
                       &buffer_loc, buffer, buffer_len);
    }

    delete [] buffer;

    if (fail){
      delete [] elem_nums;
      delete [] elem_comp;
      delete [] elem_con;
      delete [] elem_con_ptr;
      MPI_Abort(comm, fail);
      return fail;
    }

    // Arg sort the list of nodes
    int * node_args = new int[ num_nodes ]; 
    for ( int k = 0; k < num_nodes; k++ ){ 
      node_args[k] = k; 
    }

    arg_sort_list = node_nums;
    qsort(node_args, num_nodes, sizeof(int), compare_arg_sort);
    arg_sort_list = NULL;

    // Arg sort the list of elements
    int * elem_args = new int[ num_elements ];
    for ( int k = 0; k < num_elements; k++ ){
      elem_args[k] = k;
    }

    arg_sort_list = elem_nums;
    qsort(elem_args, num_elements, sizeof(int), compare_arg_sort);
    arg_sort_list = NULL;

    // Now node_nums[node_args[k]] and elem_nums[elem_args[k] are sorted.

    // Create the output for the nodes
    Xpts = new double[3*num_nodes];

    for ( int k = 0; k < num_nodes; k++ ){
      int n = node_args[k];
      for ( int j = 0; j < 3; j++ ){
        Xpts[3*k+j] = Xpts_unsorted[3*n+j];
      }
    }

    // Now handle the connectivity array
    elem_node_con = new int[ elem_con_size ];
    elem_node_ptr = new int[ num_elements+1 ];
    elem_component = new int[ num_elements ];

    elem_node_ptr[0] = 0;
    for ( int k = 0, n = 0; k < num_elements; k++ ){
      int e = elem_args[k];

      for ( int j = elem_con_ptr[e]; j < elem_con_ptr[e+1]; j++, n++ ){
        int node_num = elem_con[j];

        // Find node_num in the list
        int node = find_index_arg_sorted(node_num, num_nodes,
                                         node_nums, node_args);
        if (node < 0){
          elem_node_con[n] = -1;
          fail = 1;
        }
        else {
          elem_node_con[n] = node;
        }
      }

      elem_component[k] = elem_comp[e];
      elem_node_ptr[k+1] = n;
    }

    // Find the boundary condition nodes
    for ( int k = 0; k < num_bcs; k++ ){
      int node = find_index_arg_sorted(bc_nodes[k], num_nodes,
                                       node_nums, node_args);    
      if (node < 0){
        fail = 1;
        bc_nodes[k] = -1;
      }
      else {
        bc_nodes[k] = node;
      }
    }

    delete [] elem_nums;
    delete [] elem_comp;
    delete [] elem_con;
    delete [] elem_con_ptr;
    delete [] elem_args;

    delete [] node_args;
  }

  // Distribute the component numbers and descritpions 
  MPI_Bcast(&num_components, 1, MPI_INT, root, comm);
  if (rank != root){
    component_elems = new char[9*num_components];
    component_descript = new char[33*num_components];
  }
  MPI_Bcast(component_elems, 9*num_components, MPI_CHAR, root, comm);
  MPI_Bcast(component_descript, 33*num_components, MPI_CHAR, root, comm);

  elements = new TACSElement*[ num_components ];
  for ( int k = 0; k < num_components; k++ ){
    elements[k] = NULL;
  }

  // Broadcast the boundary condition information
  MPI_Bcast(&num_bcs, 1, MPI_INT, root, comm);

  if (rank != root){
    bc_nodes = new int[ num_bcs ];
    bc_ptr = new int[ num_bcs+1 ];
  }

  MPI_Bcast(bc_nodes, num_bcs, MPI_INT, root, comm);
  MPI_Bcast(bc_ptr, num_bcs+1, MPI_INT, root, comm);

  // Allocate the values and offsets
  int bc_size = bc_ptr[num_bcs];
  if (rank != root){
    bc_vals = new double[bc_size];
    bc_con = new int[bc_size];
  }

  // Save the original boundary condition information on the root proc
  // such that it will be possible to write the original BDF file
  if (rank == root){
    orig_bc_nodes = new int[ num_bcs ];
    orig_bc_ptr = new int[ num_bcs+1 ];
    orig_bc_vals = new double[bc_size];
    orig_bc_con = new int[bc_size];
    memcpy(orig_bc_nodes, bc_nodes, num_bcs*sizeof(int));
    memcpy(orig_bc_ptr, bc_ptr, (num_bcs+1)*sizeof(int));
    memcpy(orig_bc_vals, bc_vals, bc_size*sizeof(double));
    memcpy(orig_bc_con, bc_con, bc_size*sizeof(int));
  }

  MPI_Bcast(bc_vals, bc_size, MPI_DOUBLE, root, comm);
  MPI_Bcast(bc_con, bc_size, MPI_INT, root, comm);

  return fail;
}

/*
  Retrieve the number of nodes in the model
*/
int TACSMeshLoader::getNumNodes(){
  return num_nodes;
}

/*
  Split the mesh into segments for parallel computations.

  First, build an element to element CSR data structure. Next, split
  the mesh using Metis ane return the elem_partition array.
*/
void TACSMeshLoader::splitMesh( int split_size, 
                                int * elem_partition, int * new_nodes,
                                int * owned_elements, 
				int * owned_nodes ){
  // Create an element -> element CSR data structure for splitting with
  // Metis
    
  // First, compute the node to element CSR data structure
  // node_elem_con[node_elem_ptr[node]:node_elem_ptr[node+1]] are the
  // elements that contain node, 'node'
  int * node_elem_ptr = new int[ num_nodes+1 ];
  memset(node_elem_ptr, 0, (num_nodes+1)*sizeof(int));
    
  for ( int i = 0; i < num_elements; i++ ){
    int end = elem_node_ptr[i+1]; 
    for ( int j = elem_node_ptr[i]; j < end; j++ ){
      int node = elem_node_con[j];
      node_elem_ptr[node+1]++;
    }
  }

  // Determine the size of the node to element array
  for ( int i = 0; i < num_nodes; i++ ){
    node_elem_ptr[i+1] += node_elem_ptr[i];
  }
  int * node_elem_con = new int[ node_elem_ptr[num_nodes] ];

  for ( int i = 0; i < num_elements; i++ ){
    int end = elem_node_ptr[i+1];
    for ( int j = elem_node_ptr[i]; j < end; j++ ){
      int node = elem_node_con[j];
      node_elem_con[node_elem_ptr[node]] = i;
      node_elem_ptr[node]++;
    }
  }
  
  // Reset the node_elem_ptr array to the correct range
  for ( int i = num_nodes; i > 0; i-- ){
    node_elem_ptr[i] = node_elem_ptr[i-1];
  }
  node_elem_ptr[0] = 0;

  // Set up the element to element connectivity.
  // For this to work, must remove the diagonal contribution
  // (no self-reference)
  int * elem_ptr = new int[ num_elements+1 ];
  elem_ptr[0] = 0;

  // Information to keep track of how big the data structure is
  int elem_con_size = 0;
  int max_elem_con_size = 27*num_elements;
  int * elem_con = new int[ max_elem_con_size ];

  int * row = new int[ num_elements ];

  for ( int i = 0; i < num_elements; i++ ){
    int row_size = 0;    

    // Add the element -> element connectivity
    for ( int j = elem_node_ptr[i]; j < elem_node_ptr[i+1]; j++ ){
      int node = elem_node_con[j];

      int start = node_elem_ptr[node];
      int size = node_elem_ptr[node+1] - start;

      row_size = FElibrary::mergeArrays(row, row_size, 
                                        &node_elem_con[start], size);
    }

    if ( elem_con_size + row_size > max_elem_con_size ){
      max_elem_con_size += 0.5*max_elem_con_size;
      if (max_elem_con_size < elem_con_size + row_size ){
        max_elem_con_size += elem_con_size + row_size;
      }
      extend_int_array(&elem_con, elem_con_size, 
                       max_elem_con_size);
    }

    // Add the elements - minus the diagonal entry
    for ( int j = 0; j < row_size; j++ ){
      if (row[j] != i){ // Not the diagonal 
        elem_con[elem_con_size] = row[j];
        elem_con_size++;
      }
    }
    elem_ptr[i+1] = elem_con_size;
  }

  delete [] row;
  delete [] node_elem_ptr;
  delete [] node_elem_con;

  // Partition the mesh
  if (split_size > 1 ){
    int options[5];
    options[0] = 0; // use the default options
    int wgtflag = 0; // weights are on the verticies
    int numflag = 0; // C style numbering 
    int edgecut = -1;        
      
    int * vwgts = NULL; // Weights on the vertices 
    int * adjwgts = NULL;  // Weights on the edges or adjacency
      
    if (split_size < 8){
      METIS_PartGraphRecursive(&num_elements, elem_ptr, elem_con, 
                               vwgts, adjwgts, 
                               &wgtflag, &numflag, &split_size, 
                               options, &edgecut, elem_partition);
    }
    else {
      METIS_PartGraphKway(&num_elements, elem_ptr, elem_con, 
                          vwgts, adjwgts, 
                          &wgtflag, &numflag, &split_size, 
                          options, &edgecut, elem_partition);
    }
  }
  else {
    for ( int k = 0; k < num_elements; k++ ){
      elem_partition[k] = 0;
    }
  } 

  delete [] elem_con;
  delete [] elem_ptr;

  // Now, re-order the variables so that they are almost contiguous over
  // each processor 
  memset(owned_nodes, 0, split_size*sizeof(int));
  memset(owned_elements, 0, split_size*sizeof(int));

  // Set up the array new_nodes such that
  // old node i -> new node new_nodes[i]
  for ( int k = 0; k < num_nodes; k++ ){
    new_nodes[k] = -1;
  }

  int count = 0;
  for ( int j = 0; j < num_elements; j++ ){
    int owner = elem_partition[j];
    owned_elements[owner]++;

    for ( int i = elem_node_ptr[j]; i < elem_node_ptr[j+1]; i++ ){
      // elem_node_con[j]
      int node = elem_node_con[i];
      if (new_nodes[node] < 0){
        new_nodes[node] = count;
        owned_nodes[owner]++;
        count++;
      }
    }
  }

  // Now, number them for real
  int * split_offset = new int[ split_size ];
  split_offset[0] = 0;
  for ( int k = 1; k < split_size; k++ ){
    split_offset[k] = split_offset[k-1] + owned_nodes[k-1];
  }

  for ( int k = 0; k < num_nodes; k++ ){
    new_nodes[k] = -1;
  }

  for ( int j = 0; j < num_elements; j++ ){
    int owner = elem_partition[j];

    for ( int i = elem_node_ptr[j]; i < elem_node_ptr[j+1]; i++ ){
      // elem_node_con[j]
      int node = elem_node_con[i];
      if (new_nodes[node] < 0){
        new_nodes[node] = split_offset[owner];
        split_offset[owner]++;
      }
    }
  }

  delete [] split_offset;
}

/*
  Create a TACSToFH5 file creation object
*/
TACSToFH5 * TACSMeshLoader::createTACSToFH5( TACSAssembler * tacs,
                                             enum ElementType elem_type,
                                             unsigned int write_flag ){
  // Set the component numbers in the elements
  for ( int k = 0; k < num_components; k++ ){
    elements[k]->setComponentNum(k);
  }

  TACSToFH5 * f5 = new TACSToFH5(tacs, elem_type, write_flag);

  for ( int k = 0; k < num_components; k++ ){
    if (strlen(&component_descript[33*k]) == 0){
      char name[64];
      sprintf(name, "Component %d", k);
      f5->setComponentName(k, name);
    }
    else {
      f5->setComponentName(k, &component_descript[33*k]);
    }
  }

  return f5;
}

/*
  Create a distributed version of TACS
*/
TACSAssembler * TACSMeshLoader::createTACS( int vars_per_node,
					    int num_load_cases,
					    enum TACSAssembler::OrderingType order_type, 
					    enum TACSAssembler::MatrixOrderingType mat_type ){
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  num_owned_elements = 0; 
  int num_owned_nodes = 0;
  int *partition = NULL, *new_nodes = NULL;
  int *owned_elements = NULL, *owned_nodes = NULL; // The arrays on the root

  int root = 0;
  if (rank == root){   
    partition = new int[ num_elements ];
    new_nodes = new int[ num_nodes ];
    owned_elements = new int[size];
    owned_nodes = new int[size];
    
    splitMesh(size, partition, new_nodes, 
              owned_elements, owned_nodes);    
  }
  
  MPI_Scatter(owned_nodes, 1, MPI_INT, 
              &num_owned_nodes, 1, MPI_INT, root, comm);
  MPI_Scatter(owned_elements, 1, MPI_INT, 
              &num_owned_elements, 1, MPI_INT, root, comm);

  int num_local_nodes;
  // Local element connectivity information
  local_component_nums = new int[num_owned_elements]; // Store for use later
  int * local_elem_node_ptr = new int[num_owned_elements+1];
  int * local_elem_node_con = NULL;

  // Loacal nodal information
  int * local_tacs_nodes = NULL;
  double * Xpts_local = NULL;

  // For each processor, send the information to the owner
  if (rank == root){
    // Reset the nodes for the boundary conditions
    for ( int j = 0; j < num_bcs; j++ ){
      bc_nodes[j] = new_nodes[bc_nodes[j]];
    }

    // The element partition
    int * elem_part = new int[ num_elements ];
    for ( int k = 0; k < num_elements; k++ ){
      elem_part[k] = k;
    }

    int * inv_new_nodes = new int[ num_nodes ];
    for ( int i = 0; i < num_nodes; i++ ){
      inv_new_nodes[new_nodes[i]] = i;
    }

    // Now partition[elem_part[k]] is sorted
    arg_sort_list = partition;
    qsort(elem_part, num_elements, sizeof(int), compare_arg_sort);
    arg_sort_list = NULL;

    // Compute the local CSR data structure on this process
    // Use an upper bound for the memory requirements
    int * tacs_nodes = new int[ elem_node_ptr[num_elements] ];
    int * elem_ptr = new int[ num_elements+1 ];
    int * elem_con = new int[ elem_node_ptr[num_elements] ];
    int * elem_comp = new int[ num_elements ];
    double * xpts = new double[ 3*num_nodes ];

    int start = 0;
    for ( int k = 0; k < size; k++ ){
      int end = start + owned_elements[k];

      int local_node_size = 0;
      elem_ptr[0] = 0;

      // Cycle through partition k
      int n = 0;
      for ( int j = start; j < end; j++, n++ ){
        int elem = elem_part[j];

	// Add the element
        for ( int i = elem_node_ptr[elem]; 
	      i < elem_node_ptr[elem+1]; i++, local_node_size++ ){
          int node = elem_node_con[i];
          tacs_nodes[local_node_size] = new_nodes[node];
	  elem_con[local_node_size] = new_nodes[node];          
        }

	elem_comp[n] = elem_component[elem];
	elem_ptr[n+1] = local_node_size;
      }

      // Uniquify the list
      local_node_size = FElibrary::uniqueSort(tacs_nodes, local_node_size);

      // tacs_nodes is sorted and defines the local ordering 
      // tacs_nodes[i] -> local node i      
      for ( int j = 0; j < elem_ptr[n]; j++ ){
	int * item = (int*)bsearch(&elem_con[j], tacs_nodes, local_node_size,
				   sizeof(int), FElibrary::comparator);
	if (item){
	  elem_con[j] = item - tacs_nodes;
	}
	else {
	  fprintf(stderr, 
		  "[%d] TACSMeshLoader: could not find %d node in local list\n",
		  rank, elem_con[j]);
	  MPI_Abort(comm, 1);
	}
      }

      // Copy over the data from the local node numbers
      for ( int j = 0; j < local_node_size; j++ ){
	int node = inv_new_nodes[tacs_nodes[j]];
	xpts[3*j] = Xpts[3*node];
	xpts[3*j+1] = Xpts[3*node+1];
	xpts[3*j+2] = Xpts[3*node+2];
      }
    
      // Create the CSR data structure required
      if (k == root){
	// Copy the values over from the nodes
	num_local_nodes = local_node_size;
	local_tacs_nodes = new int[ num_local_nodes ];
	Xpts_local = new double[ 3*num_local_nodes ];
	
	memcpy(local_tacs_nodes, tacs_nodes, num_local_nodes*sizeof(int));
	memcpy(Xpts_local, xpts, 3*num_local_nodes*sizeof(double));

	// Copy over values for the elements
	memcpy(local_component_nums, elem_comp, 
	       num_owned_elements*sizeof(int));
	memcpy(local_elem_node_ptr, elem_ptr, 
	       (num_owned_elements+1)*sizeof(int));

	local_elem_node_con = new int[local_elem_node_ptr[num_owned_elements]];
	memcpy(local_elem_node_con, elem_con,
	       local_elem_node_ptr[num_owned_elements]*sizeof(int));
      }
      else {
        // Send the data to the other process
	MPI_Send(&local_node_size, 1, MPI_INT, k, 1, comm);

	MPI_Send(tacs_nodes, local_node_size, MPI_INT, k, 2, comm);
	MPI_Send(xpts, 3*local_node_size, MPI_DOUBLE, k, 3, comm);

	// Send the element data
        MPI_Send(elem_comp, owned_elements[k], MPI_INT, k, 4, comm);
        MPI_Send(elem_ptr, owned_elements[k]+1, MPI_INT, k, 5, comm);

        MPI_Send(elem_con, elem_ptr[owned_elements[k]], MPI_INT, k, 6, comm);
      }

      start += owned_elements[k];
    }

    delete [] elem_part;
    delete [] inv_new_nodes;
    delete [] tacs_nodes;
    delete [] elem_ptr;
    delete [] elem_con;
    delete [] elem_comp;
    delete [] xpts;
  }
  else {
    // Recv the data from the root process
    MPI_Status status;
    MPI_Recv(&num_local_nodes, 1, MPI_INT, 
	     root, 1, comm, &status);

    // Allocate space for the incoming data
    local_tacs_nodes = new int[ num_local_nodes ];
    Xpts_local = new double[ 3*num_local_nodes ];

    MPI_Recv(local_tacs_nodes, num_local_nodes, MPI_INT, 
	     root, 2, comm, &status);
    MPI_Recv(Xpts_local, 3*num_local_nodes, MPI_DOUBLE, 
	     root, 3, comm, &status);

    // Receive the element data
    MPI_Recv(local_component_nums, num_owned_elements, MPI_INT, 
	     root, 4, comm, &status);
    MPI_Recv(local_elem_node_ptr, num_owned_elements+1, MPI_INT, 
	     root, 5, comm, &status);

    int con_size = local_elem_node_ptr[num_owned_elements];
    local_elem_node_con = new int[con_size];
    MPI_Recv(local_elem_node_con, con_size, MPI_INT, 
	     root, 6, comm, &status);
  }
  
  int node_max_csr_size = local_elem_node_ptr[num_owned_elements];  
  TACSAssembler * tacs = new TACSAssembler(comm, num_owned_nodes, vars_per_node,
                                           num_owned_elements, num_local_nodes,
                                           node_max_csr_size, num_load_cases);
  // Sort out the boundary conditions
  // Broadcast the boundary condition information
  MPI_Bcast(bc_nodes, num_bcs, MPI_INT, root, comm);

  // Get the local node numbers for the boundary conditions
  for ( int k = 0; k < num_bcs; k++ ){
    int * item = (int*)bsearch(&bc_nodes[k], local_tacs_nodes, num_local_nodes, 
			       sizeof(int), FElibrary::comparator);
    if (item){
      bc_nodes[k] = item - local_tacs_nodes;
    }
    else {
      bc_nodes[k] = -1;
    }
  }

  // Add the node numbers - this steals the reference
  tacs->addNodes(&local_tacs_nodes);
  
  // Add the elements
  for ( int k = 0; k < num_owned_elements; k++ ){
    TACSElement * element = elements[local_component_nums[k]];
    if (!element){
      fprintf(stderr, 
              "[%d] TACSMeshLoader: Element undefined for component %d\n",
              rank, local_component_nums[k]);
      MPI_Abort(comm, 1);
      return NULL;
    }

    // Add the element node numbers
    int start = local_elem_node_ptr[k];
    int end = local_elem_node_ptr[k+1];
    tacs->addElement(element, &local_elem_node_con[start], end-start);
  }

  tacs->computeReordering(order_type, mat_type);

  // Finalize the ordering
  tacs->finalize();

  // Set the nodes
  TacsScalar * x;
  tacs->getNodeArray(&x);
  
  for ( int k = 0; k < 3*num_local_nodes; k++ ){
    x[k] = Xpts_local[k];
  }

  // Set the boundar conditions
  int bvars[6];
  TacsScalar bvals[6];
  for ( int k = 0; k < num_bcs; k++ ){
    if (bc_nodes[k] >= 0){
      int nbcs = bc_ptr[k+1] - bc_ptr[k];
      int n = 0;
      for ( int j = 0; j < nbcs; j++ ){
        if (bc_con[bc_ptr[k] + j] < vars_per_node){
          bvars[n] = bc_con[bc_ptr[k] + j];
          bvals[n] = bc_vals[bc_ptr[k] + j];
          n++;
        }
      }
      if (n > 0){
        tacs->addBC(bc_nodes[k], bvars, bvals, n);
      }
    }
  }
  
  if (rank == root){
    delete [] new_nodes;
    delete [] owned_elements;
    delete [] owned_nodes;
  }

  delete [] partition;
  delete [] local_elem_node_ptr;
  delete [] local_elem_node_con;
  delete [] Xpts_local;

  return tacs;
}

/*
  Given the split size, create a serial version of TACS
*/
TACSAssembler * TACSMeshLoader::createSerialTACS( int split_size,
                                                  int vars_per_node,
                                                  int num_load_cases ){
  // Partition the problem
  int * partition = new int[ num_elements ];
  int * new_nodes = new int[ num_nodes ];
  int * owned_elements = new int[ split_size ];
  int * owned_nodes = new int[ split_size ];
  
  splitMesh(split_size, partition, new_nodes, 
            owned_elements, owned_nodes);

  int node_max_csr_size = elem_node_ptr[num_elements];
  TACSAssembler * tacs = new TACSAssembler(MPI_COMM_SELF, 
                                           num_nodes, vars_per_node,
                                           num_elements, num_nodes,
                                           node_max_csr_size, num_load_cases);
  tacs->addNodes(&new_nodes);
  
  // Add the elements
  for ( int k = 0; k < num_elements; k++ ){
    TACSElement * element = elements[elem_component[k]];
    if (!element){
      int rank;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr, 
              "[%d] TACSMeshLoader: Element undefined for component %d\n",
              rank, elem_component[k]);
      MPI_Abort(comm, 1);
      return NULL;
    }

    // Add the element node numbers
    int start = elem_node_ptr[k];
    int end = elem_node_ptr[k+1];
    tacs->addElement(element, &elem_node_con[start], end-start);
  }

  // Finalize the ordering
  tacs->finalize();

  // Set the nodes
  TacsScalar * x;
  tacs->getNodeArray(&x);
  
  for ( int k = 0; k < 3*num_nodes; k++ ){
    x[k] = Xpts[k];
  }

  // Set the boundary condition information
  int bvars[6];
  TacsScalar bvals[6];
  for ( int k = 0; k < num_bcs; k++ ){
    int nbcs = bc_ptr[k+1] - bc_ptr[k];
    int n = 0;
    for ( int j = 0; j < nbcs; j++ ){
      if (bc_con[bc_ptr[k] + j] < vars_per_node){
        bvars[n] = bc_con[bc_ptr[k] + j];
        bvals[n] = bc_vals[bc_ptr[k] + j];
        n++;
      }
    }
    if (n > 0){
      tacs->addBC(bc_nodes[k], bvars, bvals, n);
    }
  }

  delete [] partition;
  delete [] owned_elements;
  delete [] owned_nodes;

  return tacs;
}

/*
  Retrieve the number of elements owned by this processes
*/
int TACSMeshLoader::getNumElements(){
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (!local_component_nums){
    fprintf(stderr, "[%d] TACSMeshLoader: Cannot retrieve the number \
of elements until the mesh is partitioned\n", rank);
    return 0;
  }

  return num_owned_elements;
}

/*
  Retrieve the array of elements corresponding to the
*/
void TACSMeshLoader::getComponentNums( int comp_nums[], int num_elements ){
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (!local_component_nums){
    fprintf(stderr, "[%d] TACSMeshLoader: Cannot retrieve the component numbers \
until the mesh is partitioned\n", rank);
    return;
  }

  int size = (num_owned_elements < num_elements ? num_owned_elements :
              num_elements);

  memcpy(comp_nums, local_component_nums, size*sizeof(int));
}

/*
  Retrieve the element numbers on each processor corresponding to
  the given component numbers.
*/
int TACSMeshLoader::getComponentElementNums( int ** elem_nums,
                                             int comp_nums[], int num_comps ){
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (!local_component_nums){
    fprintf(stderr, "[%d] TACSMeshLoader: Cannot get elements until \
mesh is partitioned\n", rank);
    *elem_nums = NULL;
    return 0;
  }

  // Sort the component numbers on input
  qsort(comp_nums, num_comps, sizeof(int), FElibrary::comparator);

  int * all_elems = new int[ num_owned_elements ];
  int elem_size = 0;
  for ( int k = 0; k < num_owned_elements; k++ ){
    if (bsearch(&local_component_nums[k], comp_nums, num_comps,
                sizeof(int), FElibrary::comparator)){
      all_elems[elem_size] = k;
      elem_size++;
    }
  }

  *elem_nums = new int[ elem_size ];
  memcpy(*elem_nums, all_elems, elem_size*sizeof(int));

  delete [] all_elems;

  return elem_size;
}

/*
  Set the function domain

  Given the function, and the set of component numbers that define
  the domain of interest, set the element numbers in the function that
*/
void TACSMeshLoader::setFunctionDomain( TACSFunction * function,
					int comp_nums[], int num_comps ){
  int *elems;
  int num_elems = getComponentElementNums(&elems, comp_nums, num_comps);
  function->setDomain(elems, num_elems);
  delete [] elems;
}

/*
  Determine the number of elements that are in each component
*/
void TACSMeshLoader::getNumElementsForComps( int * numElem, 
					     int sizeNumComp ){
  memset(numElem, 0, sizeof(int)*num_components);
    
  int rank;
  int root =0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    for (int i=0; i<num_elements; i++){
      if (elem_component[i] < sizeNumComp){
	numElem[elem_component[i]]++;
      }
    }
  }
}

/*
  Return the total number of elements
*/
int TACSMeshLoader::getTotalNumElements(){
  return num_elements;
}

/*
  Return the equilivent second-order connectivity of the entire mesh
*/
void TACSMeshLoader::getConnectivity( int* conn, 
				      int sizeConn ){

  memset(conn, 0, sizeof(int)*sizeConn);
    
  int rank;
  int root =0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    int ii=0;
    for ( int i = 0; i < num_elements; i++ ){
      int end = elem_node_ptr[i+1];

      int j = elem_node_ptr[i];
      // Extract only the linear connectivity...ignore mid side nodes
      if (end - j == 4){
	conn[ii  ] = elem_node_con[j  ];
	conn[ii+1] = elem_node_con[j+1];
	conn[ii+2] = elem_node_con[j+3];
	conn[ii+3] = elem_node_con[j+2];
	ii += 4;
      }
      else if (end - j == 9){
	conn[ii  ] = elem_node_con[j  ];
	conn[ii+1] = elem_node_con[j+2];
	conn[ii+2] = elem_node_con[j+8];
	conn[ii+3] = elem_node_con[j+6];
	ii += 4;
      }
      else if (end - j == 16){
	conn[ii  ] = elem_node_con[j  ];
	conn[ii+1] = elem_node_con[j+3];
	conn[ii+2] = elem_node_con[j+15];
	conn[ii+3] = elem_node_con[j+12];
	ii += 4;
      }
      else {
	conn[ii  ] = -1;
	conn[ii+1] = -1;
	conn[ii+2] = -1;
	conn[ii+3] = -1;
	ii += 4;
      }
    }
  }
}

/* 
   Return the component number for each element
*/
void TACSMeshLoader::getElementComponents( int* compIDs, 
					   int sizeCompIDs ){
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0){
    for ( int i = 0; i < num_elements; i++ ){
      compIDs[i] = elem_component[i];
    }
  }
}

/* 
   Return the second-order connectivities for elements in component
   'compID'. This is similar to getConnectivity() above.
*/
void TACSMeshLoader::getConnectivityForComp( int compID, int * conn, 
					     int sizeConn ){
  memset(conn, 0, sizeof(int)*sizeConn);
    
  int rank;
  int root = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    int ii = 0;
    for ( int i = 0; i < num_elements; i++ ){
      if (elem_component[i] == compID){
	int end = elem_node_ptr[i+1];
	int j = elem_node_ptr[i];
	// Extract only the linear connectivity...ignore mid side nodes
	if (end - j == 4){
	  conn[ii  ] = elem_node_con[j  ];
	  conn[ii+1] = elem_node_con[j+1];
	  conn[ii+2] = elem_node_con[j+3];
	  conn[ii+3] = elem_node_con[j+2];
	  ii += 4;
	}
	else if (end - j == 9){
 	  conn[ii  ] = elem_node_con[j  ];
	  conn[ii+1] = elem_node_con[j+2];
	  conn[ii+2] = elem_node_con[j+8];
	  conn[ii+3] = elem_node_con[j+6];
	  ii += 4;
	}
	else if (end - j == 16){
 	  conn[ii  ] = elem_node_con[j  ];
	  conn[ii+1] = elem_node_con[j+3];
	  conn[ii+2] = elem_node_con[j+15];
	  conn[ii+3] = elem_node_con[j+12];
	  ii += 4;
	}
      }
    }
  }
}

/*
  This function returns the nodes specified by the indices in nodeList
*/
void TACSMeshLoader::getNodes( int * nodeList, int nNodes, 
			       double * pts, int nPts ){
  int rank;
  int root = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    int ii = 0;
    for ( int i = 0; i < nNodes; i++ ) {
      pts[3*ii  ] = Xpts[3*nodeList[i]  ];
      pts[3*ii+1] = Xpts[3*nodeList[i]+1];
      pts[3*ii+2] = Xpts[3*nodeList[i]+2];
      ii++; 
    }
  }
}

/*
  This function returns original nodes in the BDF ordering
*/
void TACSMeshLoader::getOrigNodes( double * xOrig, int n){
  int rank;
  int root =0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    memcpy(xOrig, Xpts_unsorted, n*sizeof(double));
  }
}

/*
  This function returns original node numbers in the BDF ordering
*/
void TACSMeshLoader::getOrigNodeNums( int * nodeNumsOrig, int n){
  int rank;
  int root =0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root){
    memcpy(nodeNumsOrig, node_nums, n*sizeof(int));
  }
}

/*
  Create a static string that is the prefix for the NASTRAN file
*/
static const char nastran_file_header[] =
  "$ Generated by TACS - NASTRAN interface\n\
$ Nastran input desk\n\
SOL 101\n\
TIME 99999\n\
CEND\n\
ECHO = NONE\n\
STRAIN = ALL\n\
SPC = 1\n\
STRESS = ALL\n\
LOAD = 1\n\
DISP = ALL\n\
$\n\
BEGIN BULK\n\
$\n\
PARAM,AUTOSPC,NO\n\
PARAM,GRDPNT,1\n\
PARAM,K6ROT,1e5\n\
PARAM,OGEOM,YES\n\
PARAM,COUPMASS,-1\n\
PARAM,MAXRATIO,1.0e7\n\
PARAM,POST,-1\n\
PARAM,WTMASS,1.0\n";

/*
  Write a BDF file with the material properties taken from the
  TACSAssembler object 
*/
void TACSMeshLoader::writeBDF( const char * fileName, TacsScalar * bdfNodes, 
			       int * nodeNums, int nBDFNodes ){
  int size;
  MPI_Comm_size(comm, &size);

  if (size > 1){
    fprintf(stderr, "TACSMeshLoader::writeBDF() only works in serial\n");
    return;
  }

  FILE *fp = fopen(fileName, "w");
  if (fp){
    fprintf(fp, nastran_file_header);

    //  ----------- Write the Grid Nodes -----------
    fprintf(fp, "$       grid data              0\n");
    int coordDisp = 0;
    int coordId = 0;
    int seid = 0;
    for ( int i = 0; i < nBDFNodes/3; i++){
      fprintf(fp, "%8s%16d%16d%16.9f%16.9f*%7d\n","GRID*   ", 
	      nodeNums[i], coordId,
	      RealPart(bdfNodes[3*i]), 
	      RealPart(bdfNodes[3*i+1]), i+1);
      fprintf(fp, "*%7d%16.9f%16d%16s%16d        \n", 
	      i+1, RealPart(bdfNodes[3*i+2]), 
	      coordDisp, " ", seid);
    }

    // ================================================== 
    // WARNING: THE FOLLOWING ELEMENT WRITING IS HORRENDOUSLY
    // INEFFICIENT: WE LOOP OVER ALL ELEMENTS FOR EACH COMPONENT
    // DESCRIPTION SELECTING ONLY THE ONES WE NEED. FOR LARGE NUMBERS OF
    // COMPONENTS THIS YIELDS A n^2 LOOP.  CONSIDER DOING A INVERSE
    // MAPPING THAT SHOULD BE MORE EFFICIENT.
    // ==================================================
    
    //  ----------- Write the Descriptions------
    for ( int ii = 0; ii < num_components; ii++ ){
      fprintf(fp, "%-41s","$       Shell element data for family");
      fprintf(fp, "%-33s\n", &component_descript[33*ii]);
    
      //  ----------- Write the Elements ----------
      for ( int i = 0; i < num_elements; i++ ){
	if (elem_component[i] == ii) {
	  int end = elem_node_ptr[i+1];
	  int j = elem_node_ptr[i];
	  int partId = elem_component[i]+1;
	  if (end - j == 4){
	    fprintf(fp, "%-8s%8d%8d%8d%8d%8d%8d%8d\n", "CQUADR", 
		    i+1, partId, nodeNums[elem_node_con[j]],
		    nodeNums[elem_node_con[j+1]], 
		    nodeNums[elem_node_con[j+3]], 
		    nodeNums[elem_node_con[j+2]], partId);
	  }
	  else if (end - j == 9){
	    fprintf(fp, "%-8s%8d%8d%8d%8d%8d%8d%8d%8d\n", "CQUAD", 
		    i+1, partId, nodeNums[elem_node_con[j]],
		    nodeNums[elem_node_con[j+2]], 
		    nodeNums[elem_node_con[j+8]], 
		    nodeNums[elem_node_con[j+6]],
		    nodeNums[elem_node_con[j+1]], 
		    nodeNums[elem_node_con[j+5]]);
	    fprintf(fp, "%-8s%8d%8d%8d%8d\n", " ", 
		    nodeNums[elem_node_con[j+7]], 
		    nodeNums[elem_node_con[j+3]],
		    nodeNums[elem_node_con[j+4]], partId);
	  }
	  else if (end -j == 2){
	    /*
	    TACSElement * elem = elements[ii];
	    EBBeam * beam = dynamic_cast<EBBeam*>(elem);
	    if (beam){
	      enum EBBeamReferenceDirection ref_dir_type = beam->getRefDirType();
	      const TacsScalar * ref_dir = beam->getRefAxis();
	      		
	      if (ref_dir_type == STRONG_AXIS){
		printf("ERROR: bdf output does not work for beams with STRONG_AXIS specified.\n");
		fclose(fp);
		return;
	      }
	      fprintf(fp, "%-8s%8d%8d%8d%8d%8f%8f%8f\n", "CBAR", 
		      i+1, partId, nodeNums[elem_node_con[j]],
		      nodeNums[elem_node_con[j+1]], 
		      RealPart(ref_dir[0]), 
		      RealPart(ref_dir[1]), 
		      RealPart(ref_dir[2]));
	    }
	    */
	  }
	}
      }
    }
    
    //  ----------- Write the Constraints -------
    fprintf(fp, "$       Single point Constraint\n");
    char bcString[8];

    for ( int k = 0; k < num_bcs; k++ ){
      // Print node number
      fprintf(fp, "%8s%8d%8d", "SPC     ", 1, nodeNums[orig_bc_nodes[k]]);
      
      // Clear BC String
      for ( int j = 0; j < 8; j++ ){
	bcString[j] = ' ';
      }

      int kk = 0;
      for ( int j = orig_bc_ptr[k]; j < orig_bc_ptr[k+1]; j++ ){
	if (orig_bc_con[j] == 0)
	  bcString[7-kk] = '1';
	if (orig_bc_con[j] == 1)
	  bcString[7-kk] = '2';
	if (orig_bc_con[j] == 2)
	  bcString[7-kk] = '3';
	if (orig_bc_con[j] == 3)
	  bcString[7-kk] = '4';
	if (orig_bc_con[j] == 4)
	  bcString[7-kk] = '5';
	if (orig_bc_con[j] == 5)
	  bcString[7-kk] = '6';
	kk += 1;
      }

      // Print bc string and the value
      fprintf(fp, "%8s%8.6f\n", bcString, orig_bc_vals[orig_bc_ptr[k]]);
    }

    writeBDFConstitutive(fp); 
    
    // Signal end of bulk data section and close file handle
    fprintf(fp, "ENDDATA\n");
    fclose(fp);
  }
}

/*
  Write the constitutve data from the TACSMeshLoader class to a new
  BDF file. This does not write any nodal information or
  connectivities.

  Note this code is serial. If you try to run it in parallell then it
  will use results from just the root processor. This will work but
  may not provide the most up-to-date constitutive data since not all
  processors will modify the constitutive objects stored in
  TACSMeshLoader. This can be fixed by assigning the design variables
  individually to each element in the TACSMeshLoader class.

  input:
  filename: the file name that will be written
*/
void TACSMeshLoader::writeBDFConstitutive( const char *filename ){
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0){
    // Open the file on the root processor
    FILE *fp = fopen(filename, "w");

    if (fp){
      // Print the NASTRAN header info
      fprintf(fp, nastran_file_header);

      // Write all the constitutive data
      writeBDFConstitutive(fp);
      
      // End the data
      fprintf(fp, "ENDDATA\n");

      // Close the file
      fclose(fp);
    }
  }
}

/*
  Write the constitutive data from the TACSMeshLoader class to a bdf
  file.

  Note that this code does not close the file handle and assumes that
  it has been properly opened. This code is serial and can only be run
  on a single processor.

  input:
  fp:   an open file handle
*/
void TACSMeshLoader::writeBDFConstitutive( FILE *fp ){
  if (fp){
    // Scan through all the components and write out the constitutive data 
    // for each one as long as it is an FSDT or beam element
    int matCount = 0;
    for ( int i = 0; i < num_components; i++ ){
      int PID = i+1;

      // Get the base constitutive objective
      TACSConstitutive * con = elements[i]->getConstitutive();

      // Flag if the constitutive object is written
      int conWritten = 0;

      // Currently we can only deal with constitutive classes that
      // inherit from FSDTStiffness or EBStiffness
      
      FSDTStiffness * fcon = dynamic_cast<FSDTStiffness*>(con);
      if (fcon){

 	// Flag the constiutive object as being written
	conWritten = 1;
	
	//First Get the stiffness
	TacsScalar A[6], B[6], D[6], As[3];
	double gpt[2] = {0.0, 0.0};
	
	TacsScalar rho, mass[3];
	rho = 0.0;
	mass[0] = mass[1] = mass[2] = 0.0;

	fcon->getStiffness(gpt, A, B, D, As);
	fcon->pointwiseMass(gpt, mass);
	rho = fcon->getDensity();
      
	if (rho == 0.0){
	  printf("Warning: the FSDTconsitutive class did not implement getDensity().\
 Mass properties *WILL BE WRONG IN BDF FILE!*\n");
	  rho = 1.0;
	}

	// This should give the average thickness
	TacsScalar tfact = 1.0;
	TacsScalar t = mass[0]/rho;
	
	if (mass[2] != 0.0){
	  tfact = (12.0*mass[2])/(rho*t*t*t); 
	}
	
	// Reconstruct the ref axis information
	const TacsScalar * tmp = fcon->getRefAxis();
	TacsScalar axis[3];
	axis[0] = tmp[0];
	axis[1] = tmp[1];
	axis[2] = tmp[2];
	  
	// Check if the axis is exactly equal to 0,0,0
	if (axis[0] == 0.0 && axis[1] == 0.0 && axis[2] == 0.0){
	  //Axis is not defined so it shouldn't matter:
	  axis[0] = 1.0;
	}

	// Generate another vector that is not parallel and not
	// parallel to the axis. 
	TacsScalar v1[3], zaxis[3];
	v1[0] = 0.0;
	v1[1] = 1.0;
	v1[2] = 0.0;
	Tensor::normalize3D(v1);

	// Check if the dot product is 1.0
	TacsScalar dp = Tensor::dot3D(v1, axis);
	if (abs(dp) > 1-1e-15){
	  // Use a different axis
	  v1[0] = 1.0;
	  v1[1] = 0.0;
	  v1[2] = 0.0;
	}
	Tensor::crossProduct3D(zaxis, axis, v1);

	// Write the 4 material identifiers
	int MID1 = matCount+1;
	int MID2 = matCount+2;
	int MID3 = matCount+3;
	int MID4 = matCount+4;
	double A1 = 0.0;
	double A2 = 0.0;
	double A3 = 0.0;
	double Tref = 293.15;
	TacsScalar t2 = t*t;
	TacsScalar t3 = t*t*t*tfact;
	
	fprintf(fp, "%8s%16d%16.9e%16.9e%16.9e*\n",   "MAT2*   ", 
		MID1, RealPart(A[0]/t), RealPart(A[1]/t), RealPart(A[2]/t));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e*\n", " ", 
		RealPart(A[3]/t), RealPart(A[4]/t), RealPart(A[5]/t), 
		RealPart(rho));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e\n", " ", 
		A1, A2, A3, Tref);
	
	fprintf(fp, "%8s%16d%16.9e%16.9e%16.9e*\n",   "MAT2*   ", 
		MID2, RealPart(12*D[0]/t3), 
		RealPart(12*D[1]/t3), RealPart(12*D[2]/t3));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e*\n", " ", 
		RealPart(12*D[3]/t3), RealPart(12*D[4]/t3), 
		RealPart(12*D[5]/t3), RealPart(rho));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e\n", " ", 
		A1, A2, A3, Tref);
      
	fprintf(fp, "%8s%16d%16.9e%16.9e%16.9e*\n",   "MAT2*   ", 
		MID3, RealPart(As[0]/t), RealPart(As[1]/t), 0.0);
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e*\n", " ", 
		RealPart(As[2]/t), 0.0, 0.0, RealPart(rho));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e\n", " ", 
		A1, A2, A3, Tref);
	
	fprintf(fp, "%8s%16d%16.9e%16.9e%16.9e*\n",   "MAT2*   ", 
		MID4, -RealPart(B[0]/t2), -RealPart(B[1]/t2), 
		-RealPart(B[2]/t2));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e*\n", " ", 
		-RealPart(B[3]/t2), -RealPart(B[4]/t2), 
		-RealPart(B[5]/t2), RealPart(rho));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e%16.9e\n", " ", 
		A1, A2, A3, Tref);
	
	// And the PSHELL entry
	double NSM = 0.0;
	fprintf(fp, "%-8s", "PSHELL*");
	fprintf(fp, "%16d", PID);
	fprintf(fp, "%16d", MID1);
	fprintf(fp, "%16.9e", RealPart(t));
	fprintf(fp, "%16d*\n", MID2);
	fprintf(fp, "*%7s", " ");
	fprintf(fp, "%16.9e", RealPart(tfact));
	fprintf(fp, "%16d", MID3);
	fprintf(fp, "%16.9e", 1.0);
	fprintf(fp, "%16.9e*\n", NSM);
	fprintf(fp, "*%7s", " ");
	fprintf(fp, "%16s", " ");
	fprintf(fp, "%16s", " ");
	fprintf(fp, "%16d\n", MID4);
      
	// And finally the coordinate system.
	fprintf(fp, "%-8s", "CORD2R");
	fprintf(fp, "%8d", i+1);
	fprintf(fp, "%8d", 0);
	fprintf(fp, "%8.3f", 0.0);
	fprintf(fp, "%8.3f", 0.0);
	fprintf(fp, "%8.3f", 0.0);
	fprintf(fp, "%8.3f", RealPart(zaxis[0]));
	fprintf(fp, "%8.3f", RealPart(zaxis[1]));
	fprintf(fp, "%8.3f\n", RealPart(zaxis[2]));
	fprintf(fp, "        %8.3f%8.3f%8.3f\n",
		RealPart(axis[0]), RealPart(axis[1]), 
		RealPart(axis[2]));
	
	matCount += 4;
      }

      /*
      EBStiffness * econ = dynamic_cast<EBStiffness*>(con);
      if (econ){
	// Flag the constiutive object as being written
	conWritten = 1;

	// And the PBEAM entry
	double NSM = 0.0;
	fprintf(fp, "%-8s%16d%16d%16.9e%16.9e*\n",   "PBAR*",
		PID, matCount+1, RealPart(econ->A), 
		RealPart(econ->Iy));
	fprintf(fp, "*%7s%16.9e%16.9e%16.9e\n", " ",
		RealPart(econ->Ix), RealPart(econ->J), NSM);

	// Now write a MAT1 entry.
	fprintf(fp, "%-8s%16d%16.9e%16.9e%16s*\n",   "MAT1*",
		matCount+1, RealPart(econ->E), 
		RealPart(econ->G), " ");
	fprintf(fp, "*%7s%16.9e\n", " ", RealPart(econ->rho));
	
	matCount += 1;
      }
      */

      if (conWritten == 0){
	printf("Cannot write BDF file. Must consist of only constitutive \
objects that inherit from FSDTStiffness and EBStiffness\n");
	return;
      }
    }
  }
}
   
