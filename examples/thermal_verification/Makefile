include ../../Makefile.in
include ../../TACS_Common.mk

OBJS = create_stiff_panel.o \
	thermal_linear.o \
	thermal_nonlinear.o

default: ${OBJS}
	${CXX} -o create_stiff_panel create_stiff_panel.o ${TACS_LD_FLAGS}
	${CXX} -o thermal_linear thermal_linear.o ${TACS_LD_FLAGS}
	${CXX} -o thermal_nonlinear thermal_nonlinear.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

clean:
	rm -f *.o create_stiff_panel thermal_linear thermal_nonlinear
	rm -f *.bdf axial_stiffened_panel

test: default
	./create_stiff_panel
	./thermal_linear
	./thermal_nonlinear
	mpirun -np 4 ./stiffened_panel
	mpirun -np 4 ./thermal_linear
	mpirun -np 4 ./thermal_nonlinear

test_complex: default
	./create_stiff_panel
	mpirun -np 4 ./stiffened_panel
	mpirun -np 4 ./thermal_linear
	mpirun -np 4 ./thermal_nonlinear

create_stiff_panel: create_stiff_panel.o
	${CXX} -o create_stiff_panel create_stiff_panel.o ${TACS_LD_FLAGS}

thermal_linear: thermal_linear.o
	${CXX} -o thermal_linear thermal_linear.o ${TACS_LD_FLAGS}

thermal_nonlinear: thermal_nonlinear.o
	${CXX} -o thermal_nonlinear thermal_nonlinear.o ${TACS_LD_FLAGS}