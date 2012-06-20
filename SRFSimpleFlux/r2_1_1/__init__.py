#!/usr/bin/env python

#---------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.    If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV, Andrey SIMURZIN
##


#---------------------------------------------------------------------------
from Foam import man, ref


#---------------------------------------------------------------------------
def createFields( runTime, mesh ):

    ref.ext_Info() << "Reading field p\n" << ref.nl
    
    p = man.volScalarField( man.IOobject( ref.word( "p" ), 
                                          ref.fileName( runTime.timeName() ), 
                                          mesh, 
                                          ref.IOobject.MUST_READ, 
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )

    ref.ext_Info() << "Reading field Urel\n" << ref.nl
    Urel = man.volVectorField( man.IOobject( ref.word( "Urel" ),
                                             ref.fileName( runTime.timeName() ),
                                             mesh,
                                             ref.IOobject.MUST_READ,
                                             ref.IOobject.AUTO_WRITE ),
                               mesh );
    
    ref.ext_Info() << "Reading/calculating face flux field phi\n" << ref.nl
    phi = man.surfaceScalarField( man.IOobject( ref.word( "phi" ),
                                                ref.fileName( runTime.timeName() ),
                                                mesh,
                                                ref.IOobject.READ_IF_PRESENT,
                                                ref.IOobject.AUTO_WRITE ), 
                                  man.surfaceScalarField( ref.linearInterpolate( Urel ) & mesh.Sf(), man.Deps( mesh, Urel ) ) )
    
    pRefCell = 0
    pRefValue = 0.0

    pRefCell, pRefValue = ref.setRefCell( p, mesh.solutionDict().subDict( ref.word( "SIMPLE" ) ), pRefCell, pRefValue )

    laminarTransport = man.singlePhaseTransportModel( Urel, phi )

    turbulence = man.incompressible.RASModel.New( Urel, phi, laminarTransport )
    
    ref.ext_Info() << "Creating SRF model\n" << ref.nl
    SRF = man.SRF.SRFModel.New( Urel ) 
        
    sources = man.IObasicSourceList( mesh )
    
    return p, Urel, phi, pRefCell, pRefValue, laminarTransport, turbulence, SRF, sources


#---------------------------------------------------------------------------
def fun_UrelEqn( Urel, phi, turbulence, p, sources, SRF ):
    # Solve the Momentum equation
    
    UrelEqn = man.fvVectorMatrix( ref.fvm.div( phi, Urel ) + turbulence.divDevReff( Urel ) + SRF.Su(), man.Deps( turbulence, Urel, phi, SRF ) )    \
                 == man( sources( Urel ), man.Deps( Urel ) )

    UrelEqn.relax()

    sources.constrain( UrelEqn )

    ref.solve( UrelEqn == -man.fvc.grad( p ) )

    return UrelEqn


#---------------------------------------------------------------------------
def fun_pEqn( mesh, runTime, simple, Urel, phi, turbulence, p, UrelEqn, pRefCell, pRefValue, cumulativeContErr, sources ):
    
    p.ext_boundaryField().updateCoeffs()

    rAUrel = 1.0 / UrelEqn().A();
    Urel << rAUrel * UrelEqn().H() 
    
    phi << ( ref.fvc.interpolate( Urel, ref.word( "interpolate(HbyA)" ) ) & mesh.Sf() )
    
    ref.adjustPhi(phi, Urel, p)

    # Non-orthogonal pressure corrector loop
    while simple.correctNonOrthogonal():
        pEqn = ref.fvm.laplacian( rAUrel, p ) == ref.fvc.div( phi )

        pEqn.setReference( pRefCell, pRefValue )

        pEqn.solve()

        if simple.finalNonOrthogonalIter():
            phi -= pEqn.flux()
            pass
        pass
    cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )

    # Explicitly relax pressure for momentum corrector
    p.relax()

    # Momentum corrector
    Urel -= rAUrel * ref.fvc.grad( p )
    Urel.correctBoundaryConditions()
    
    sources.correct( Urel )
    
    return cumulativeContErr


#---------------------------------------------------------------------------
def main_standalone( argc, argv ):
    
    args = ref.setRootCase( argc, argv )
    
    runTime = man.createTime( args )
    
    mesh = man.createMesh( runTime )
        
    p, Urel, phi, pRefCell, pRefValue, laminarTransport, turbulence, SRF, sources = createFields( runTime, mesh )

    cumulativeContErr = ref.initContinuityErrs()
    
    simple = man.simpleControl (mesh)

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    ref.ext_Info() << "\nStarting time loop\n" << ref.nl

    while simple.loop():
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl

        # --- Pressure-velocity SIMPLE corrector
        UrelEqn = fun_UrelEqn( Urel, phi, turbulence, p, sources, SRF )
        cumulativeContErr = fun_pEqn( mesh, runTime, simple, Urel, phi, turbulence, p, UrelEqn, pRefCell, pRefValue, cumulativeContErr, sources )

        turbulence.correct()
        
        Uabs = None
        if runTime.outputTime():
                Uabs = ref.volVectorField( ref.IOobject( ref.word( "Uabs" ),
                                                         ref.fileName( runTime.timeName() ),
                                                         mesh,
                                                         ref.IOobject.NO_READ,
                                                         ref.IOobject.AUTO_WRITE ),
                                           Urel() + SRF.U() ) # mixed calculations
                pass

        runTime.write()

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" \
                        << "    ClockTime = " << runTime.elapsedClockTime() << " s" \
                        << ref.nl << ref.nl
        pass

    ref.ext_Info() << "End\n" << ref.nl

    import os
    return os.EX_OK


#---------------------------------------------------------------------------
from Foam import FOAM_REF_VERSION
if FOAM_REF_VERSION( ">=", "020101" ):
     if __name__ == "__main__" :
            import sys, os
            argv = sys.argv
            os._exit( main_standalone( len( argv ), argv ) )
            pass
     pass
else:
     ref.ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam2.1.1 \n "         
        


