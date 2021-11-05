include ../../Makefile.in
include ../../TACS_Common.mk

OBJS = caps.o

default: ${OBJS}
	${CXX} -o caps caps.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

clean:
	rm -f *.o caps

test: default
	./create_stiff_panel
	# ./create_skewed_plate
	./stiffened_panel_nonlinear
	./cyl_nonlinear
	mpirun -np 4 ./stiffened_panel
	# mpirun -np 4 ./skewed_plate
	mpirun -np 4 ./stiffened_panel_nonlinear
	mpirun -np 4 ./cyl_nonlinear

test_complex: default
	./create_stiff_panel
	# ./create_skewed_plate
	mpirun -np 4 ./stiffened_panel_nonlinear
	mpirun -np 4 ./stiffened_panel
	mpirun -np 4 ./cyl_nonlinear
	# mpirun -np 4 ./skewed_plate
	mpirun -np 4 ./stiffened_panel_nonlinear
	mpirun -np 4 ./cyl_nonlinear

cyl_nonlinear: cyl_nonlinear.o
	${CXX} -o cyl_nonlinear cyl_nonlinear.o ${TACS_LD_FLAGS}

create_stiff_panel: create_stiff_panel.o
	${CXX} -o create_stiff_panel create_stiff_panel.o ${TACS_LD_FLAGS}

caps: caps.o
	${CXX} -o caps caps.o ${TACS_LD_FLAGS}