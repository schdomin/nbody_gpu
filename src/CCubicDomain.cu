#include "CCubicDomain.cuh"



namespace NBody
{

//ds ctor/dtor
//ds default constructor requires environmental parameters: N number of bodies, dT time step, T number of time steps
CCubicDomain::CCubicDomain( const unsigned int& p_uNumberOfParticles ): m_arrPositions( 0 ),
                                                                        m_arrVelocities( 0 ),
                                                                        m_arrAccelerations( 0 ),
                                                                        m_arrMasses( 0 ),
                                                                        m_uNumberOfParticles( p_uNumberOfParticles ),
                                                                        m_strParticleInformation( "" ),
                                                                        m_strIntegralsInformation( "" )
{
    //ds nothing to do
}

//ds default destructor
CCubicDomain::~CCubicDomain( )
{
    //ds deallocate memory
    delete[] m_arrPositions;
    delete[] m_arrVelocities;
    delete[] m_arrAccelerations;
    delete[] m_arrMasses;
}

//ds accessors
void CCubicDomain::createParticlesUniformFromNormalDistribution( const float& p_dTargetKineticEnergy, const float& p_fParticleMass )
{
    //ds allocate arrays (linear since we're using CUDA)
    m_arrPositions     = new float[m_uNumberOfParticles*3];
    m_arrVelocities    = new float[m_uNumberOfParticles*3];
    m_arrAccelerations = new float[m_uNumberOfParticles*3];
    m_arrMasses        = new float[m_uNumberOfParticles];

    //ds kinetic energy to derive from initial situation
    float dKineticEnergy( 0.0 );

    //ds set particle information for each
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds set the particle mass (same for all particles in this case)
        m_arrMasses[u] = p_fParticleMass;

        //ds set the position: uniformly distributed between boundaries in this case
        m_arrPositions[3*u+0] = _getUniformlyDistributedNumber( );
        m_arrPositions[3*u+1] = _getUniformlyDistributedNumber( );
        m_arrPositions[3*u+2] = _getUniformlyDistributedNumber( );

        //ds set velocity values: from normal distribution
        m_arrVelocities[3*u+0] = _getNormallyDistributedNumber( );
        m_arrVelocities[3*u+1] = _getNormallyDistributedNumber( );
        m_arrVelocities[3*u+2] = _getNormallyDistributedNumber( );

        //ds set acceleration values: 0
        m_arrAccelerations[3*u+0] = 0;
        m_arrAccelerations[3*u+1] = 0;
        m_arrAccelerations[3*u+2] = 0;

        //ds add the resulting kinetic component (needed below)
        dKineticEnergy += m_arrMasses[u]/2*pow( sqrt( pow( m_arrVelocities[3*u+0], 2 )
                                                    + pow( m_arrVelocities[3*u+1], 2 )
                                                    + pow( m_arrVelocities[3*u+2], 2 ) ), 2 );
    }

    //ds calculate the rescaling factor
    const float dRescalingFactor( sqrt( p_dTargetKineticEnergy/dKineticEnergy ) );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds rescale the velocity component
        m_arrVelocities[3*u+0] = dRescalingFactor*m_arrVelocities[3*u+0];
        m_arrVelocities[3*u+1] = dRescalingFactor*m_arrVelocities[3*u+1];
        m_arrVelocities[3*u+2] = dRescalingFactor*m_arrVelocities[3*u+2];
    }
}

void CCubicDomain::saveParticlesToStream( )
{
    //ds format: X Y Z U V W

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds get a buffer for snprintf (max possible buffer size checked)
        char chBuffer[256];

        //ds get the particle stream
        std::snprintf( chBuffer, 100, "%f %f %f %f %f %f", m_arrPositions[3*u+0], m_arrPositions[3*u+1], m_arrPositions[3*u+2],
                                                           m_arrVelocities[3*u+0], m_arrVelocities[3*u+1], m_arrVelocities[3*u+2] );

        //ds append the buffer to our string
        m_strParticleInformation += chBuffer;
        m_strParticleInformation += "\n";
    }
}

void CCubicDomain::saveIntegralsToStream( const float& p_fTotalEnergy )
{
    //ds format: E X Y Z X Y Z X Y Z

    //ds buffer for snprintf
    char chBuffer[256];

    //ds get information - caution, memory gets allocated
    const float* vecCenterOfMass    = getCenterOfMass( );
    const float* vecAngularMomentum = getAngularMomentum( );
    const float* vecLinearMomentum  = getLinearMomentum( );

    //ds get the integrals stream
    std::snprintf( chBuffer, 100, "%f %f %f %f %f %f %f %f %f %f", p_fTotalEnergy,
                                                                   vecCenterOfMass[0], vecCenterOfMass[1], vecCenterOfMass[2],
                                                                   vecAngularMomentum[0], vecAngularMomentum[1], vecAngularMomentum[2],
                                                                   vecLinearMomentum[0], vecLinearMomentum[1], vecLinearMomentum[2] );

    //ds free memory
    delete vecCenterOfMass;
    delete vecAngularMomentum;
    delete vecLinearMomentum;

    //ds append the buffer to our string
    m_strIntegralsInformation += chBuffer;
    m_strIntegralsInformation += "\n";
}

void CCubicDomain::saveIntegralsToStream( const float& p_fTotalEnergy, const float p_vecCenterOfMass[3] )
{
    //ds format: E X Y Z X Y Z X Y Z

    //ds buffer for snprintf
    char chBuffer[256];

    //ds get information - caution, memory gets allocated
    //const float* vecCenterOfMass    = getCenterOfMass( );
    const float* vecAngularMomentum = getAngularMomentum( );
    const float* vecLinearMomentum  = getLinearMomentum( );

    //ds get the integrals stream
    std::snprintf( chBuffer, 100, "%f %f %f %f %f %f %f %f %f %f", p_fTotalEnergy,
                                                                   p_vecCenterOfMass[0], p_vecCenterOfMass[1], p_vecCenterOfMass[2],
                                                                   vecAngularMomentum[0], vecAngularMomentum[1], vecAngularMomentum[2],
                                                                   vecLinearMomentum[0], vecLinearMomentum[1], vecLinearMomentum[2] );

    //ds free memory
    //delete vecCenterOfMass;
    delete vecAngularMomentum;
    delete vecLinearMomentum;

    //ds append the buffer to our string
    m_strIntegralsInformation += chBuffer;
    m_strIntegralsInformation += "\n";
}

void CCubicDomain::writeParticlesToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps )
{
    //ds ofstream object
    std::ofstream ofsFile;

    //ds open the file for writing
    ofsFile.open( p_strFilename.c_str( ), std::ofstream::out );

    //ds if it worked
    if( ofsFile.is_open( ) )
    {
        //ds first dump setup information number of particles and timesteps
        ofsFile << m_uNumberOfParticles << " " << p_uNumberOfTimeSteps << "\n" << m_strParticleInformation;
    }

    //ds close the file
    ofsFile.close( );
}

void CCubicDomain::writeIntegralsToFile( const std::string& p_strFilename, const unsigned int& p_uNumberOfTimeSteps, const double& p_dTimeStepSize )
{
    //ds ofstream object
    std::ofstream ofsFile;

    //ds open the file for writing
    ofsFile.open( p_strFilename.c_str( ), std::ofstream::out );

    //ds if it worked
    if( ofsFile.is_open( ) )
    {
        //ds dump first integrals information
        ofsFile << p_uNumberOfTimeSteps << " " << p_dTimeStepSize << "\n" << m_strIntegralsInformation;
    }

    //ds close the file
    ofsFile.close( );
}

float* CCubicDomain::getPositions( )
{
    return m_arrPositions;
}

float* CCubicDomain::getVelocities( )
{
    return m_arrVelocities;
}

float* CCubicDomain::getAccelerations( )
{
    return m_arrAccelerations;
}

float* CCubicDomain::getMasses( )
{
    return m_arrMasses;
}

const float* CCubicDomain::getCenterOfMass( ) const
{
    //ds center to find
    float* vecCenter = new float[3];

    //ds set it to zero for sure
    vecCenter[0] = 0.0;
    vecCenter[1] = 0.0;
    vecCenter[2] = 0.0;

    //ds total mass
    float fMassTotal( 0.0 );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds mass instance
        const float fCurrentMass( m_arrMasses[u] );

        //ds add the current relative mass
        vecCenter[0] += fCurrentMass*m_arrPositions[3*u+0];
        vecCenter[1] += fCurrentMass*m_arrPositions[3*u+1];
        vecCenter[2] += fCurrentMass*m_arrPositions[3*u+2];

        //ds add the current mass
        fMassTotal += fCurrentMass;
    }

    //ds divide by total mass
    vecCenter[0] /= fMassTotal;
    vecCenter[1] /= fMassTotal;
    vecCenter[2] /= fMassTotal;

    return vecCenter;
}

const float* CCubicDomain::getAngularMomentum( ) const
{
    //ds momentum
    float* vecMomentum = new float[3];

    //ds set it to zero for sure
    vecMomentum[0] = 0.0;
    vecMomentum[1] = 0.0;
    vecMomentum[2] = 0.0;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds mass instance
        const float fCurrentMass( m_arrMasses[u] );

        //ds add the current momentum
        vecMomentum[0] += fCurrentMass*( m_arrPositions[3*u+1]*m_arrVelocities[3*u+2] - m_arrPositions[3*u+2]*m_arrVelocities[3*u+1] );
        vecMomentum[1] += fCurrentMass*( m_arrPositions[3*u+2]*m_arrVelocities[3*u+0] - m_arrPositions[3*u+0]*m_arrVelocities[3*u+2] );
        vecMomentum[2] += fCurrentMass*( m_arrPositions[3*u+0]*m_arrVelocities[3*u+1] - m_arrPositions[3*u+1]*m_arrVelocities[3*u+0] );
    }

    return vecMomentum;
}

const float* CCubicDomain::getLinearMomentum( ) const
{
    //ds momentum
    float* vecMomentum = new float[3];

    //ds set it to zero for sure
    vecMomentum[0] = 0.0;
    vecMomentum[1] = 0.0;
    vecMomentum[2] = 0.0;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds mass instance
        const float fCurrentMass( m_arrMasses[u] );

        //ds add the current momentum
        vecMomentum[0] += fCurrentMass*m_arrVelocities[3*u+0];
        vecMomentum[1] += fCurrentMass*m_arrVelocities[3*u+1];
        vecMomentum[2] += fCurrentMass*m_arrVelocities[3*u+2];
    }

    return vecMomentum;
}

float CCubicDomain::_getUniformlyDistributedNumber( ) const
{
    //ds drand48 returns [0,1], we need [-1,1] -> therefore 2x[0,1] -> [0,2] -> -1 ->[0-1,2-1] = [-1,1]
    return static_cast< float >( 2*drand48( )-1 );
}

float CCubicDomain::_getNormallyDistributedNumber( ) const
{
    //ds calculate the uniform number first [0,1]
    const float dUniformNumber( static_cast< float >( drand48( ) ) );

    //ds return the normal one
    return static_cast< float >( sqrt( -2*log( dUniformNumber ) )*cos( 2*static_cast< float >( M_PI )*dUniformNumber ) );
}

} //ds namespace NBody
