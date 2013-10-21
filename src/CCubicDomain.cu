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
void CCubicDomain::createParticlesUniformFromNormalDistribution( const double& p_dTargetKineticEnergy, const float& p_fParticleMass )
{
    //ds allocate arrays (linear since we're using CUDA)
    m_arrPositions     = new float[m_uNumberOfParticles*3];
    m_arrVelocities    = new float[m_uNumberOfParticles*3];
    m_arrAccelerations = new float[m_uNumberOfParticles*3];
    m_arrMasses        = new float[m_uNumberOfParticles];

    //ds kinetic energy to derive from initial situation
    double dKineticEnergy( 0.0 );

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
    const double dRescalingFactor( sqrt( p_dTargetKineticEnergy/dKineticEnergy ) );

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

void CCubicDomain::saveIntegralsToStream( const double& p_dMinimumDistance, const double& p_dPotentialDepth )
{
    /*ds format: E X Y Z X Y Z X Y Z

    //ds buffer for snprintf
    char chBuffer[256];

    //ds get the integrals stream
    std::snprintf( chBuffer, 100, "%f %f %f %f %f %f %f %f %f %f", getTotalEnergy( p_dMinimumDistance, p_dPotentialDepth ),
                                                                   getCenterOfMass( )( 0 ), getCenterOfMass( )( 1 ), getCenterOfMass( )( 2 ),
                                                                   getAngularMomentum( )( 0 ), getAngularMomentum( )( 1 ), getAngularMomentum( )( 2 ),
                                                                   getLinearMomentum( )( 0 ), getLinearMomentum( )( 1 ), getLinearMomentum( )( 2 ) );

    //ds append the buffer to our string
    m_strIntegralsInformation += chBuffer;
    m_strIntegralsInformation += "\n";*/
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

/*ds accessors/helpers
double CCubicDomain::getTotalEnergy( const double& p_dMinimumDistance, const double& p_dPotentialDepth ) const
{
    //ds total energy to accumulate
    double dTotalEnergy( 0.0 );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds loop over all other particles (dont do the same particles twice)
        for( unsigned int v = u+1; v < m_uNumberOfParticles; ++v )
        {
            //ds add the kinetic component
            dTotalEnergy += m_arrParticles[u].m_dMass/2*pow( NBody::CVector::absoluteValue( m_arrParticles[u].m_cVelocity ), 2 );

            //ds add the potential component
            dTotalEnergy += _getLennardJonesPotential( m_arrParticles[u], m_arrParticles[v], p_dMinimumDistance, p_dPotentialDepth );
        }
    }

    return dTotalEnergy;
}

CVector CCubicDomain::getCenterOfMass( ) const
{
    //ds center to find
    CVector cCenter;

    //ds total mass
    double dMassTotal( 0.0 );

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current relative mass
        cCenter += m_arrParticles[u].m_dMass*m_arrParticles[u].m_cPosition;

        //ds add the current mass
        dMassTotal += m_arrParticles[u].m_dMass;
    }

    //ds divide by total mass
    cCenter /= dMassTotal;

    return cCenter;
}

CVector CCubicDomain::getAngularMomentum( ) const
{
    //ds momentum
    CVector cMomentum;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current momentum
        cMomentum += NBody::CVector::crossProduct( m_arrParticles[u].m_cPosition, m_arrParticles[u].m_dMass*m_arrParticles[u].m_cVelocity );
    }

    return cMomentum;
}

CVector CCubicDomain::getLinearMomentum( ) const
{
    //ds momentum
    CVector cMomentum;

    //ds for each particle
    for( unsigned int u = 0; u < m_uNumberOfParticles; ++u )
    {
        //ds add the current momentum
        cMomentum += m_arrParticles[u].m_dMass*m_arrParticles[u].m_cVelocity;
    }

    return cMomentum;
}*/

//ds helpers
/*double CCubicDomain::_getLennardJonesPotential( const CParticle& p_CParticle1,  const CParticle& p_CParticle2, const double& p_dMinimumDistance, const double& p_dPotentialDepth ) const
{
    //ds formula
    return 4*p_dPotentialDepth*( pow( p_dMinimumDistance/NBody::CVector::absoluteValue( p_CParticle1.m_cPosition-p_CParticle2.m_cPosition ), 12 )
                               - pow( p_dMinimumDistance/NBody::CVector::absoluteValue( p_CParticle1.m_cPosition-p_CParticle2.m_cPosition ), 6 ) );
}*/

float CCubicDomain::_getUniformlyDistributedNumber( ) const
{
    //ds drand48 returns [0,1], we need [-1,1] -> therefore 2x[0,1] -> [0,2] -> -1 ->[0-1,2-1] = [-1,1]
    return static_cast< float >( 2*drand48( )-1 );
}

float CCubicDomain::_getNormallyDistributedNumber( ) const
{
    //ds calculate the uniform number first [0,1]
    const double dUniformNumber( drand48( ) );

    //ds return the normal one
    return static_cast< float >( sqrt( -2*log( dUniformNumber ) )*cos( 2*M_PI*dUniformNumber ) );
}

} //ds namespace NBody
