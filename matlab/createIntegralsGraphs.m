function createIntegralsGraphs( )

%ds filepath
strFilepath = '../bin/integrals.txt';

%ds open the file
fileID = fopen( strFilepath );

%ds get the first line
cCell = textscan( fileID, '%u %f', 1 );

%ds get timesteps T
uNumberOfTimesteps = cCell{1};
dTimestepSize      = cCell{2};

%ds informative
disp( ['Number of timesteps: ', num2str( uNumberOfTimesteps ) ] );
disp( ['Timestep size: ', num2str( dTimestepSize ) ] );
disp( [ 'starting data import from: ', strFilepath ] ); 
tic;

%ds get the remaining lines E X Y Z X Y Z X Y Z
cCell = textscan( fileID, '%f %f %f %f %f %f %f %f %f %f', uNumberOfTimesteps );

vecTotalEnergy           = cCell{1};
vecCenterOfMassX         = cCell{2};
vecCenterOfMassY         = cCell{3};
vecCenterOfMassZ         = cCell{4};
vecTotalAngularMomentumX = cCell{5};
vecTotalAngularMomentumY = cCell{6};
vecTotalAngularMomentumZ = cCell{7};
vecTotalLinearMomentumX  = cCell{8};
vecTotalLinearMomentumY  = cCell{9};
vecTotalLinearMomentumZ  = cCell{10};

disp( [ 'finished data import - time: ', num2str( toc ) ] );

%ds allocate the timeline
vecTimeline = zeros( uNumberOfTimesteps, 1 );

%ds current time for the vector (safety for integer/double multiplicatioo)
dCurrentTime = 0.0;

%ds fill the timeline
for u = 1:1:uNumberOfTimesteps
    
    %ds set the element
    vecTimeline( u ) = dCurrentTime;
    
    %ds update current time
    dCurrentTime = dCurrentTime+dTimestepSize;
    
end

hFigure1 = figure( 1 );
plot( vecTimeline, vecTotalEnergy );
title( 'Total energy' );
xlabel( 'Time' );
ylabel( 'Energy' );
legend( 'Total energy' );

hFigure2 = figure( 2 );
plot( vecTimeline, [ vecCenterOfMassX, vecCenterOfMassY, vecCenterOfMassZ ] );
title( 'Center of mass' );
xlabel( 'Time' );
ylabel( 'Coordinate' );
legend( 'x', 'y', 'z' );

hFigure3 = figure( 3 );
plot( vecTimeline, [ vecTotalAngularMomentumX, vecTotalAngularMomentumY, vecTotalAngularMomentumZ ] );
title( 'Total angular momentum' );
xlabel( 'Time' );
ylabel( 'Coordinate' );
legend( 'x', 'y', 'z' );

hFigure4 = figure( 4 );
plot( vecTimeline, [ vecTotalLinearMomentumX, vecTotalLinearMomentumY, vecTotalLinearMomentumZ ] );
title( 'Total linear momentum' );
xlabel( 'Time' );
ylabel( 'Coordinate' );
legend( 'x', 'y', 'z' );

disp( 'exporting figures as jpg' );

saveas( hFigure1, 'totalenergy.jpg' );
saveas( hFigure2, 'centerofmass.jpg' );
saveas( hFigure3, 'totalangularmomentum.jpg' );
saveas( hFigure4, 'totallinearmomentum.jpg' );

disp( 'function ended successfully' );

end

