#include "CCubicDomain.cuh"           //ds domain structure
#include "Timer.h"                    //ds time measurement
#include <iostream>                   //ds cout
#include <cuda.h>                     //ds needed for eclipse indexer only (not for compilation)
#include <cuda_runtime.h>             //ds needed for eclipse indexer only (not for compilation)
#include <device_launch_parameters.h> //ds needed for eclipse indexer only (not for compilation)



//ds CUDA kernels - split up acceleration and velocity verlet for better readability - no shared memory used within these blocks (no overhead due to copying of 2d arrays)
//-------------------------------------------------------------------------------------------------------------------------//
__global__ void computeAccelerationsLennardJones( const unsigned int p_uNumberOfParticles,
                                                  float* p_arrPositions,
                                                  float* p_arrMasses,
                                                  float* p_arrNewAcclerations,
                                                  const float p_fLowerBoundary,
                                                  const float p_fUpperBoundary,
                                                  const float p_fMinimumDistance,
                                                  const float p_fPotentialDepth );

__global__ void updateParticlesVelocityVerlet( const unsigned int p_uNumberOfParticles,
                                               float* p_arrPositions,
                                               float* p_arrVelocities,
                                               float* p_arrAccelerations,
                                               float* p_arrNewAcclerations,
                                               const float p_fLowerBoundary,
                                               const float p_fUpperBoundary,
                                               const float p_fTimeStepSize );

__global__ void getTotalEnergy( const unsigned int p_uNumberOfParticles,
                                float* p_arrPositions,
                                float* p_arrVelocities,
                                float* p_arrMasses,
                                const float p_fMinimumDistance,
                                const float p_fPotentialDepth,
                                float* p_fTotalEnergy );
//-------------------------------------------------------------------------------------------------------------------------//

//ds NOT USED CUDA kernels due to worse perfomance than CPU solution
/*-------------------------------------------------------------------------------------------------------------------------//
__global__ void getCenterOfMass( const unsigned int p_uNumberOfParticles,
                                 float* p_arrPositions,
                                 float* p_arrMasses,
                                 float* p_vecCenterOfMass );

__global__ void getTotalAngularMomentum( const unsigned int p_uNumberOfParticles,
                                         float* p_arrPositions,
                                         float* p_arrVelocities,
                                         float* p_arrMasses,
                                         float* p_vecTotalAngularMomentum );
//-------------------------------------------------------------------------------------------------------------------------*/

int main( int argc, char** argv )
{
    //ds start timing
    Timer tmTimer; tmTimer.start( );

    //ds domain configuration
    const std::pair< float, float > pairBoundaries( -1, 1 );
    const unsigned int uNumberOfParticles( 100 );

    //ds allocate a domain to work with specifying number of particles and timing
    NBody::CCubicDomain cDomain( uNumberOfParticles );

    //ds target kinetic energy
    const float fTargetKineticEnergy( 1000.0 );

    //ds create particles uniformly from a normal distribution - no CUDA call here
    cDomain.createParticlesUniformFromNormalDistribution( fTargetKineticEnergy );

    //ds host information: particles
    float* h_arrPositions    ( cDomain.getPositions( ) );
    float* h_arrVelocities   ( cDomain.getVelocities( ) );
    float* h_arrAccelerations( cDomain.getAccelerations( ) );
    float* h_arrMasses       ( cDomain.getMasses( ) );

    //ds host information: integrals and initialize them
    float h_fTotalEnergy( 0.0 );
    //float h_vecCenterOfMass[3];         h_vecCenterOfMass[0]         = 0.0; h_vecCenterOfMass[1]         = 0.0; h_vecCenterOfMass[2]         = 0.0;
    //float h_vecTotalAngularMomentum[3]; h_vecTotalAngularMomentum[0] = 0.0; h_vecTotalAngularMomentum[1] = 0.0; h_vecTotalAngularMomentum[2] = 0.0;

    //ds device handles: particles
    float* d_arrPositions       ( 0 ); //Nx3
    float* d_arrVelocities      ( 0 ); //Nx3
    float* d_arrAccelerations   ( 0 ); //Nx3
    float* d_arrMasses          ( 0 ); //Nx3
    float* d_arrNewAccelerations( 0 ); //Nx3

    //ds device handles: integrals
    float* d_fTotalEnergy           ( 0 ); //1x1
    //float* d_vecCenterOfMass        ( 0 ); //3x1
    //float* d_vecTotalAngularMomentum( 0 ); //3x1

    //ds allocate memory: particles (here we see the advantage of using linear arrays)
    cudaMalloc( (void **)&d_arrPositions       , uNumberOfParticles*3*sizeof( float ) );
    cudaMalloc( (void **)&d_arrVelocities      , uNumberOfParticles*3*sizeof( float ) );
    cudaMalloc( (void **)&d_arrAccelerations   , uNumberOfParticles*3*sizeof( float ) ) ;
    cudaMalloc( (void **)&d_arrMasses          , uNumberOfParticles*sizeof( float ) ) ;
    cudaMalloc( (void **)&d_arrNewAccelerations, uNumberOfParticles*3*sizeof( float ) ) ;

    //ds allocate memory: integrals
    cudaMalloc( (void **)&d_fTotalEnergy           , sizeof( float ) ) ;
    //cudaMalloc( (void **)&d_vecCenterOfMass        , 3*sizeof( float ) ) ;
    //cudaMalloc( (void **)&d_vecTotalAngularMomentum, 3*sizeof( float ) ) ;

    //ds copy memory to gpu to initialize the situation
    cudaMemcpy( d_arrPositions    , h_arrPositions    , uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrVelocities   , h_arrVelocities   , uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrAccelerations, h_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_arrMasses       , h_arrMasses       , uNumberOfParticles*sizeof( float )  , cudaMemcpyHostToDevice );

    //ds current simulation configuration
    const float fTimeStepSize            ( 0.0001 );
    const unsigned int uNumberOfTimeSteps( 5000 );
    const float fMinimumDistance         ( 0.05 );
    const float fPotentialDepth          ( 0.01 );

    //ds start simulation
    for( unsigned int uCurrentTimeStep = 0; uCurrentTimeStep < uNumberOfTimeSteps; ++uCurrentTimeStep )
    {
        //ds calculate the new accelerations
        computeAccelerationsLennardJones<<< 1, uNumberOfParticles >>>( uNumberOfParticles,
                                                                       d_arrPositions,
                                                                       d_arrMasses,
                                                                       d_arrNewAccelerations,
                                                                       pairBoundaries.first,
                                                                       pairBoundaries.second,
                                                                       fMinimumDistance,
                                                                       fPotentialDepth );

        //ds update particle properties according to velocity verlet scheme
        updateParticlesVelocityVerlet<<< 1, uNumberOfParticles >>>( uNumberOfParticles,
                                                                    d_arrPositions,
                                                                    d_arrVelocities,
                                                                    d_arrAccelerations,
                                                                    d_arrNewAccelerations,
                                                                    pairBoundaries.first,
                                                                    pairBoundaries.second,
                                                                    fTimeStepSize );

        //ds compute total energy
        getTotalEnergy<<< 1, uNumberOfParticles, uNumberOfParticles*sizeof( float )  >>>( uNumberOfParticles,
                                                                                          d_arrPositions,
                                                                                          d_arrVelocities,
                                                                                          d_arrMasses,
                                                                                          fMinimumDistance,
                                                                                          fPotentialDepth,
                                                                                          d_fTotalEnergy );

        /*ds compute center of mass
        getCenterOfMass<<< 1, uNumberOfParticles, uNumberOfParticles*4*sizeof( float ) >>>( uNumberOfParticles,
                                                                                            d_arrPositions,
                                                                                            d_arrMasses,
                                                                                            d_vecCenterOfMass );*/

        /*ds compute total angular momentum - INFO: slower than cpu version
        getTotalAngularMomentum<<< 1, uNumberOfParticles, uNumberOfParticles*3*sizeof( float ) >>>( uNumberOfParticles,
                                                                                                    d_arrPositions,
                                                                                                    d_arrVelocities,
                                                                                                    d_arrMasses,
                                                                                                    d_vecTotalAngularMomentum );*/

        //ds get the particle information from gpu to cpu
        cudaMemcpy( h_arrPositions    , d_arrPositions    , uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrVelocities   , d_arrVelocities   , uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrAccelerations, d_arrAccelerations, uNumberOfParticles*3*sizeof( float ), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_arrMasses       , d_arrMasses       , uNumberOfParticles*sizeof( float )  , cudaMemcpyDeviceToHost );

        //ds get the integrals information from gpu to cpu
        cudaMemcpy( &h_fTotalEnergy          , d_fTotalEnergy           , sizeof( float ), cudaMemcpyDeviceToHost );
        //cudaMemcpy( h_vecCenterOfMass        , d_vecCenterOfMass        , 3*sizeof( float ), cudaMemcpyDeviceToHost );
        //cudaMemcpy( h_vecTotalAngularMomentum, d_vecTotalAngularMomentum, 3*sizeof( float ), cudaMemcpyDeviceToHost );

        //ds save particle and integral information - the correct integrals saving procedure gets called automatically depending on parameters
        cDomain.saveParticlesToStream( );
        cDomain.saveIntegralsToStream( h_fTotalEnergy ); //<- only total energy resource is taken from CUDA computation
    }

    //ds deallocate memory
    cudaFree( d_arrPositions );
    cudaFree( d_arrVelocities );
    cudaFree( d_arrAccelerations );
    cudaFree( d_arrMasses );
    cudaFree( d_arrNewAccelerations );
    cudaFree( d_fTotalEnergy );
    //cudaFree( d_vecCenterOfMass );
    //cudaFree( d_vecTotalAngularMomentum );

    //ds save the streams to a file
    cDomain.writeParticlesToFile( "bin/simulation.txt", uNumberOfTimeSteps );
    cDomain.writeIntegralsToFile( "bin/integrals.txt", uNumberOfTimeSteps, fTimeStepSize );

    //ds stop timing
    const double dDurationSeconds( tmTimer.stop( ) );

    std::cout << "-------GPU SETUP------------------------------------------------------------" << std::endl;
    std::cout << "  Number of particles: " << uNumberOfParticles << std::endl;
    std::cout << "Target kinetic energy: " << fTargetKineticEnergy << std::endl;
    std::cout << "  Number of timesteps: " << uNumberOfTimeSteps << std::endl;
    std::cout << "     Computation time: " << dDurationSeconds << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    return 0;
}

//ds CUDA kernels - split up acceleration and velocity verlet for better readability - no shared memory used within these blocks (no overhead due to copying of 2d arrays)
//-------------------------------------------------------------------------------------------------------------------------//
__global__ void computeAccelerationsLennardJones( const unsigned int p_uNumberOfParticles,
                                                  float* p_arrPositions,
                                                  float* p_arrMasses,
                                                  float* p_arrNewAccelerations,
                                                  const float p_fLowerBoundary,
                                                  const float p_fUpperBoundary,
                                                  const float p_fMinimumDistance,
                                                  const float p_fPotentialDepth )
{
    //ds regular index and "real" particle index equals three times thread index, since were working with a linear 2d array
    const unsigned int uIndex1D( threadIdx.x );
    const unsigned int uIndex3D( 3*threadIdx.x );

    //ds get current mass (constant)
    const float fCurrentMass( p_arrMasses[uIndex1D] );

    //ds force instance to calculate for the current particle
    float vecTotalForce[3];

    //ds make sure all elements are initialized correctly
    vecTotalForce[0] = 0.0;
    vecTotalForce[1] = 0.0;
    vecTotalForce[2] = 0.0;

    //ds get the domain size
    const float fDomainSize( fabs( p_fLowerBoundary ) + fabs( p_fUpperBoundary ) );

    //ds loop over all other particles
    for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
    {
        //ds do not treat itself (else nan results because division by zero)
        if( u != uIndex1D )
        {
            //ds cutoff distance
            const float fDistanceCutoff = 2.5*p_fMinimumDistance;

            //ds we have to loop over the cubic boundary conditions
            for( float dX = p_fLowerBoundary; dX < p_fUpperBoundary+1.0; ++dX )
            {
                for( float dY = p_fLowerBoundary; dY < p_fUpperBoundary+1.0; ++dY )
                {
                    for( float dZ = p_fLowerBoundary; dZ < p_fUpperBoundary+1.0; ++dZ )
                    {
                        //ds get the radial vector between the particles
                        float vecRadius[3];

                        //ds calculate the distance: domain + particle2 - particle1
                        vecRadius[0] = dX*fDomainSize + p_arrPositions[3*u+0] - p_arrPositions[uIndex3D+0];
                        vecRadius[1] = dY*fDomainSize + p_arrPositions[3*u+1] - p_arrPositions[uIndex3D+1];
                        vecRadius[2] = dZ*fDomainSize + p_arrPositions[3*u+2] - p_arrPositions[uIndex3D+2];

                        //ds get the absolute distance
                        const float fDistanceAbsolute( sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) ) );

                        //ds if we are within the cutoff range (only smaller here to avoid double overhead for >=)
                        if( fDistanceCutoff > fDistanceAbsolute )
                        {
                            //ds calculate the lennard jones force prefix
                            const float fLJFPrefix( -24*p_fPotentialDepth*( 2*pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6  ) )
                                                                         *1/pow( fDistanceAbsolute, 2 ) );

                            //ds add the information to the force including the radial component
                            vecTotalForce[0] += fLJFPrefix*vecRadius[0];
                            vecTotalForce[1] += fLJFPrefix*vecRadius[1];
                            vecTotalForce[2] += fLJFPrefix*vecRadius[2];
                        }
                    }
                }
            }
        }
    }

    //ds set the new acceleration
    p_arrNewAccelerations[uIndex3D+0] = vecTotalForce[0]/fCurrentMass;
    p_arrNewAccelerations[uIndex3D+1] = vecTotalForce[1]/fCurrentMass;
    p_arrNewAccelerations[uIndex3D+2] = vecTotalForce[2]/fCurrentMass;
}

__global__ void updateParticlesVelocityVerlet( const unsigned int p_uNumberOfParticles,
                                               float* p_arrPositions,
                                               float* p_arrVelocities,
                                               float* p_arrAccelerations,
                                               float* p_arrNewAccelerations,
                                               const float p_fLowerBoundary,
                                               const float p_fUpperBoundary,
                                               const float p_fTimeStepSize )
{
    //ds 3d index for the linear array
    const unsigned int uIndex3D( 3*threadIdx.x );

    //ds calculate domain size
    const float fDomainSize( abs( p_fLowerBoundary ) + abs( p_fUpperBoundary ) );

    //ds velocity-verlet for position
    p_arrPositions[uIndex3D+0] = p_arrPositions[uIndex3D+0] + p_fTimeStepSize*p_arrVelocities[uIndex3D+0] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndex3D+0];
    p_arrPositions[uIndex3D+1] = p_arrPositions[uIndex3D+1] + p_fTimeStepSize*p_arrVelocities[uIndex3D+1] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndex3D+1];
    p_arrPositions[uIndex3D+2] = p_arrPositions[uIndex3D+2] + p_fTimeStepSize*p_arrVelocities[uIndex3D+2] + ( 1/2 )*pow( p_fTimeStepSize, 2 )*p_arrAccelerations[uIndex3D+2];

    //ds produce periodic boundary shifting - check each element: x,y,z
    for( unsigned int v = 0; v < 3; ++v )
    {
        //ds check if we are below the boundary
        while( p_fLowerBoundary > p_arrPositions[uIndex3D+v] )
        {
            //ds map the particle to the other boundary by shifting it up to the boundary
            p_arrPositions[uIndex3D+v] += fDomainSize;
        }

        //ds check if we are above the boundary
        while( p_fUpperBoundary < p_arrPositions[uIndex3D+v] )
        {
            //ds map the particle to the other boundary by shifting it back to the boundary
            p_arrPositions[uIndex3D+v] -= fDomainSize;
        }
    }

    //ds velocity-verlet for velocity
    p_arrVelocities[uIndex3D+0] = p_arrVelocities[uIndex3D+0] + ( p_fTimeStepSize/2 )*( p_arrNewAccelerations[uIndex3D+0] + p_arrAccelerations[uIndex3D+0] );
    p_arrVelocities[uIndex3D+1] = p_arrVelocities[uIndex3D+1] + ( p_fTimeStepSize/2 )*( p_arrNewAccelerations[uIndex3D+1] + p_arrAccelerations[uIndex3D+1] );
    p_arrVelocities[uIndex3D+2] = p_arrVelocities[uIndex3D+2] + ( p_fTimeStepSize/2 )*( p_arrNewAccelerations[uIndex3D+2] + p_arrAccelerations[uIndex3D+2] );

    //ds update the old accelerations
    p_arrAccelerations[uIndex3D+0] = p_arrNewAccelerations[uIndex3D+0];
    p_arrAccelerations[uIndex3D+1] = p_arrNewAccelerations[uIndex3D+1];
    p_arrAccelerations[uIndex3D+2] = p_arrNewAccelerations[uIndex3D+2];
}

__global__ void getTotalEnergy( const unsigned int p_uNumberOfParticles,
                                float* p_arrPositions,
                                float* p_arrVelocities,
                                float* p_arrMasses,
                                const float p_fMinimumDistance,
                                const float p_fPotentialDepth,
                                float* p_fTotalEnergy )
{
    //ds dynamic shared total energy to sum up by first thread
    extern __shared__ float s_arrTotalEnergy[];

    //ds regular index and "real" particle index equals three times thread index, since were working with a linear 2d array
    const unsigned int uIndex1D( threadIdx.x );
    const unsigned int uIndex3D( 3*threadIdx.x );

    //ds make sure the shared memory is empty (each thread does this)
    s_arrTotalEnergy[uIndex1D] = 0.0;

    //ds wait until all threads are done
    __syncthreads( );

    //ds calculate the total energy of the new configuration - loop over all other particles (dont do the same particles twice)
    for( unsigned int u = uIndex1D+1; u < p_uNumberOfParticles; ++u )
    {
        //ds add the kinetic component from the other particle
        s_arrTotalEnergy[uIndex1D] += p_arrMasses[u]/2*pow( sqrt( pow( p_arrVelocities[3*u+0], 2 ) + pow( p_arrVelocities[3*u+1], 2 ) + pow( p_arrVelocities[3*u+2], 2 ) ), 2 );

        //ds get the radial vector between the particles
        float vecRadius[3];

        //ds calculate the distance: particle2 - particle1
        vecRadius[0] = p_arrPositions[3*u+0] - p_arrPositions[uIndex3D+0];
        vecRadius[1] = p_arrPositions[3*u+1] - p_arrPositions[uIndex3D+1];
        vecRadius[2] = p_arrPositions[3*u+2] - p_arrPositions[uIndex3D+2];

        //ds get the absolute distance
        const float fDistanceAbsolute( sqrt( pow( vecRadius[0], 2 ) + pow( vecRadius[1], 2 ) + pow( vecRadius[2], 2 ) ) );

        //ds add the potential component
        s_arrTotalEnergy[uIndex1D] += 4*p_fPotentialDepth*( pow( p_fMinimumDistance/fDistanceAbsolute, 12 ) - pow( p_fMinimumDistance/fDistanceAbsolute, 6 ) );
    }

    //ds wait until all threads are done
    __syncthreads( );

    //ds thread 0 calculates the total energy
    if( 0 == uIndex1D )
    {
        //ds total energy to sum up
        float fTotalEnergy( 0.0 );

        for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
        {
            fTotalEnergy += s_arrTotalEnergy[u];
        }

        //ds set the return value
        *p_fTotalEnergy = fTotalEnergy;
    }
}
//-------------------------------------------------------------------------------------------------------------------------//

//ds NOT USED CUDA kernels due to worse perfomance than CPU solution
/*-------------------------------------------------------------------------------------------------------------------------//
__global__ void getCenterOfMass( const unsigned int p_uNumberOfParticles,
                                 float* p_arrPositions,
                                 float* p_arrMasses,
                                 float* p_vecCenterOfMass )
{
    //ds dynamic shared relative center of mass to sum up by first thread  + the mass (Nx4)
    extern __shared__ float s_arrRelativeCenterOfMassPlusMass[];

    //ds Nx4 Array in this case
    const unsigned int uIndex1D( threadIdx.x );
    const unsigned int uIndex4D( 4*threadIdx.x );

    //ds save current mass
    const float fCurrentMass( p_arrMasses[uIndex1D] );

    //ds set the relative mass
    s_arrRelativeCenterOfMassPlusMass[uIndex4D+0] = fCurrentMass*p_arrPositions[uIndex4D+0];
    s_arrRelativeCenterOfMassPlusMass[uIndex4D+1] = fCurrentMass*p_arrPositions[uIndex4D+1];
    s_arrRelativeCenterOfMassPlusMass[uIndex4D+2] = fCurrentMass*p_arrPositions[uIndex4D+2];

    //ds save it to the shared array too
    s_arrRelativeCenterOfMassPlusMass[uIndex4D+3] = fCurrentMass;

    //ds wait until all threads are done
    __syncthreads( );

    //ds the first thread now calculates the result
    if( 0 == uIndex1D )
    {
        //ds initialize
        p_vecCenterOfMass[0] = 0.0;
        p_vecCenterOfMass[1] = 0.0;
        p_vecCenterOfMass[2] = 0.0;

        //ds total mass
        float fTotalMass( 0.0 );

        //ds for each particle
        for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
        {
            //ds update the center
            p_vecCenterOfMass[0] += s_arrRelativeCenterOfMassPlusMass[4*u+0];
            p_vecCenterOfMass[1] += s_arrRelativeCenterOfMassPlusMass[4*u+1];
            p_vecCenterOfMass[2] += s_arrRelativeCenterOfMassPlusMass[4*u+2];

            //ds update the total mass
            fTotalMass += s_arrRelativeCenterOfMassPlusMass[4*u+3];
        }

        //ds calculate the result
        p_vecCenterOfMass[0] /= fTotalMass;
        p_vecCenterOfMass[1] /= fTotalMass;
        p_vecCenterOfMass[2] /= fTotalMass;
    }
}

__global__ void getTotalAngularMomentum( const unsigned int p_uNumberOfParticles,
                                         float* p_arrPositions,
                                         float* p_arrVelocities,
                                         float* p_arrMasses,
                                         float* p_vecTotalAngularMomentum )
{
    //ds dynamic shared memory to calculate the total angular momentum
    extern __shared__ float s_arrAngularMomentum[];

    //ds Nx3 Array in this case
    const unsigned int uIndex1D( threadIdx.x );
    const unsigned int uIndex3D( 3*threadIdx.x );

    //ds save current mass
    const float fCurrentMass( p_arrMasses[uIndex1D] );

    //ds set the relative mass
    s_arrAngularMomentum[uIndex3D+0] = fCurrentMass*( p_arrPositions[uIndex3D+1]*p_arrVelocities[uIndex3D+2] - p_arrPositions[uIndex3D+2]*p_arrVelocities[uIndex3D+1] );
    s_arrAngularMomentum[uIndex3D+1] = fCurrentMass*( p_arrPositions[uIndex3D+2]*p_arrVelocities[uIndex3D+0] - p_arrPositions[uIndex3D+0]*p_arrVelocities[uIndex3D+2] );
    s_arrAngularMomentum[uIndex3D+2] = fCurrentMass*( p_arrPositions[uIndex3D+0]*p_arrVelocities[uIndex3D+1] - p_arrPositions[uIndex3D+1]*p_arrVelocities[uIndex3D+0] );

    //ds wait until all threads are done
    __syncthreads( );

    //ds first thread does the accumulation
    if( 0 == uIndex1D )
    {
        //ds initialization
        p_vecTotalAngularMomentum[0] = 0.0;
        p_vecTotalAngularMomentum[1] = 0.0;
        p_vecTotalAngularMomentum[2] = 0.0;

        //ds loop over all particles
        for( unsigned int u = 0; u < p_uNumberOfParticles; ++u )
        {
            //ds get the values from the shared memory
            p_vecTotalAngularMomentum[0] += s_arrAngularMomentum[3*u+0];
            p_vecTotalAngularMomentum[1] += s_arrAngularMomentum[3*u+1];
            p_vecTotalAngularMomentum[2] += s_arrAngularMomentum[3*u+2];
        }
    }
}
//-------------------------------------------------------------------------------------------------------------------------*/
