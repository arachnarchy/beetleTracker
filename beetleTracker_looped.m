% This is Daniel's beetle tracker version 1.2 (Mar 7 2013)

% This is the looped version of the beetle tracker, meant for long analysis
% sessions. To get started, set current folder to the one where the videos
% are in, and create a list of file names.

videoList = dir;
videoList = {videoList.name};
videoList = videoList';
movIdx = strfind(videoList,'.mov');
movIdx = find(not(cellfun('isempty',movIdx)));
videoList = videoList(movIdx);
save('videoList');

% Checks if this is a new analysis loop or a previous one was aborted
% prematurely

if exist('it', 'var');
    % then do nothing (i.e. keep current it)
else
it = 7; % If coming back to a begun analysis, look up row of desired movie in the videoList and enter here.
end

while it <= length(videoList);

clearvars -except it videoList newScaleFactor
close all

% movies have to be in 240x240px format (bigger slows down performance
% dramatically)

%filename = 'B_C4_2_2_240px.mov'; % enter movie file name here (incl. file extension)
filename = videoList{it,1};
trunkName = filename(1:end-4);

% Display first frame and delineate zones of interest
mov = VideoReader(filename);
firstFrame=read(mov,1);
fig = figure('name',filename);
imshow(firstFrame,'InitialMagnification', 200)
title('4 mouse clicks to delineate ROI (top to bottom), 2 to mark left and right crop');
[x,y] = ginput(6);

heightLateralTop = y(1,:);
heightLateralBottom = y(2,:);
heightDorsalTop = y(3,:);
heightDorsalBottom = y(4,:);
cropLeft = x(5,:);
cropRight = x(6,:);

% This sets variables to tweak blob detection, as well as telling the
% script whether the beetle is going left to right or right to left
prompt = {'Directionality (1 for left, -1 for right', 'Theshold Scale Factor:','Minimum Blob Size', 'Maximum Blob Size'};
dlg_title = 'Inputs';
num_lines = 1;
def = {'1','0.6','20','1000'}; % Good starting values
answer = inputdlg(prompt,dlg_title,num_lines,def);

directionality = answer{1,1}; directionality = str2double(directionality);
scaleFactor = answer{2,1}; scaleFactor = str2double(scaleFactor);
minBlob = answer{3,1}; minBlob = str2double(minBlob); minBlob = uint32(minBlob);
maxBlob = answer{4,1}; maxBlob = str2double(maxBlob); maxBlob = uint32(maxBlob);

close(fig);

% Display first frame and delineate zones of interest
fig = figure
imshow(firstFrame,'InitialMagnification', 200)
title('6 clicks to mark obstacle (L-R;L-R;L-R)');
[x,y] = ginput(6);

obstacleLleft = x(1,:);
obstacleLright = x(2,:);
obstacleDleft = x(3,:);
obstacleDright = x(4,:);
obstacleDBottomLeft = x(5,:);
obstacleDBottomRight = x(6,:);

% set px to mm calibration 
obstacleLengthPx = y(5,:) - y(3,:);
obstacleLength = 68; % in mm
calib = obstacleLength/obstacleLengthPx;
 
close(fig);

% Create main object
videoFReader = vision.VideoFileReader(filename,'ImageColorSpace', 'Intensity');

% Create System objects to display videos
sz = get(0,'ScreenSize');
pos = [20 sz(4)-300 240 240];
videoOrig = vision.VideoPlayer('Name', 'Original', 'Position', pos);
pos(1) = pos(1)+260; % move the next viewer to the right
videoThresh = vision.VideoPlayer('Name', 'Threshold', 'Position', pos);
pos(1) = pos(1)+260;
videoTrack = vision.VideoPlayer('Name', 'Tracking', 'Position', pos);

% Create System object for thresholding original movie. ScaleFactor needs
% to be adjusted, 0.8 is good start value. For dark videos, try 0.7 first.
thresh = vision.Autothresholder('Operator','<=','ThresholdScaleFactor',scaleFactor); 

%Create a blob analysis System object to find potential beetles    
blob = vision.BlobAnalysis( ...
                    'AreaOutputPort', true, ...
                    'BoundingBoxOutputPort', true, ...
                    'CentroidOutputPort', true, ...
                    'MajorAxisLengthOutputPort', true, ...
                    'MinorAxisLengthOutputPort', false, ...
                    'OrientationOutputPort', true, ...
                    'MinimumBlobArea', minBlob, ...
                    'MaximumBlobArea', maxBlob, ...
                    'MaximumCount', 5);               
                
% Create System object for drawing the bounding boxes around detected objects.
blobBoxes = vision.ShapeInserter('BorderColor','white');

% Create System Object for inserting lines indicating blob orientation
orientationLines = vision.ShapeInserter('Shape','Lines','BorderColor','white');
                
% Create System object for drawing lines indicating regions of interest.
lineDorsalTop = vision.ShapeInserter('Shape','Lines','BorderColor','white');

% For plotting regions of interest
dorsalROI = [
    cropLeft 0 cropLeft 240;
    0 heightLateralTop 240 heightLateralTop;
    0 heightLateralBottom 240 heightLateralBottom;
    0 heightDorsalTop 240 heightDorsalTop; 
    0 heightDorsalBottom 240 heightDorsalBottom;
    cropRight 0 cropRight 240;];

% prepare empty matrices 
centroids = zeros(0,2);
bounds = zeros(0,4);
orientations = zeros(0,1);
frames = zeros(0,1);
objects = zeros(0,1);
majoraxes = zeros(0,1);
areas = zeros(0,1);

%%
% Start processing loop. This loop uses the previously instantiated System objects.
frameID = 1;
while ~isDone(videoFReader)
    
    videoFrame = step(videoFReader); % Read input video frame
    threshFrame = step(thresh,videoFrame); % threshold the frame
    
    % Estimate the area and bounding box of the blobs.
     [Area, Centroid, BBox, MajorAxis, Orientation] = step(blob, threshFrame);
   
     
     % this gets rid of everything outside the ROIs 
     Idx = find(BBox(:,1) < cropLeft);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     Idx = find(BBox(:,1) > cropRight);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     Idx = find(BBox(:,2) < heightLateralTop);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     Idx = find(BBox(:,2) < heightDorsalTop & BBox(:,2) > heightLateralBottom);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     Idx = find(BBox(:,2) > heightDorsalBottom);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     
     % Gets rid of all data where only one view is recognized, in order to make later corrections possible 
     objectCount = length(BBox(:,1)); % counts objects in each frame
     numberOfObjects = ones(objectCount,1) * objectCount;
     Idx = find(numberOfObjects ~=2);
     BBox(Idx,:) = [];
     Centroid(Idx,:) = [];
     Orientation(Idx) = [];
     MajorAxis(Idx) = [];
     Area(Idx) = [];
     
     % Inverts angles if run starts on the right side
     Orientation = Orientation .* directionality;
     
     % This will write each frame's data to combination variables
     objectCount = length(BBox(:,1)); % counts objects in each frame
     numberOfObjects = ones(objectCount,1) * objectCount; % writes a column with the object counts so it lines up with the data rows
     frame = ones(objectCount,1) * frameID; % writes a column with frameIDs so it lines up with the data rows
     
     % this successively builds matrices for the whole movie
     bounds = [bounds;BBox]; 
     centroids = [centroids;Centroid];
     orientations = [orientations;Orientation];
     frames = [frames; frame];
     objects = [objects; numberOfObjects];
     majoraxes = [majoraxes;MajorAxis];
     areas = [areas;Area];
     
     % this calculates orientation lines (computation intensive, only use
     % for verification)
%         lineLength = 50;
% 
%         pointX = Centroid(:,1);
%         pointY = Centroid(:,2);
% 
%         pointMinusX = pointX - lineLength * cos(Orientation);
%         pointMinusY = pointY + lineLength * sin(Orientation);
% 
%         pointPlusX = pointX + lineLength * cos(Orientation);
%         pointPlusY = pointY - lineLength * sin(Orientation);
% 
%         angles = [pointMinusX, pointMinusY, pointX, pointY, pointPlusX, pointPlusY];
%      
     
    % Draws bounding rectangles, the ROI, and orienation lines
    blobOverlay = step(blobBoxes, videoFrame, BBox); % adds bounding boxes
    blobOverlay = step(lineDorsalTop, blobOverlay,dorsalROI); % adds ROI marker lines
    %blobOverlay = step(orientationLines, blobOverlay, angles); % adds orientation marker lines
    
    step(videoOrig, videoFrame); % Display original video
    step(videoThresh,threshFrame); % Display thresholded video
    step(videoTrack,blobOverlay); % Display original video with bounding boxes and ROI limits
   
    frameID = frameID+1;
end

%%
 % split into lateral and dorsal
     lateralIdx = find(bounds(:,2) > heightLateralTop & bounds(:,2) < heightLateralBottom);
     dorsalIdx = find(bounds(:,2) > heightDorsalTop & bounds(:,2) < heightDorsalBottom);
     
     % convert pixels to mm
     bounds = bounds .* calib;
     centroids = centroids .* calib;
     majoraxes = majoraxes .* calib;
     areas = areas .* calib;
     
     bboxL = bounds(lateralIdx,:);
     bboxD = bounds(dorsalIdx,:);
     centroidL = centroids(lateralIdx,:);
     centroidD = centroids(dorsalIdx,:);
     areasL = areas(lateralIdx,:);
     areasD = areas(dorsalIdx,:);
     orientationL = orientations(lateralIdx,:);
     orientationD = orientations(dorsalIdx,:);
     objectsL = objects(lateralIdx,:);
     objectsD = objects(dorsalIdx,:);
     framesL = frames(lateralIdx,:);
     framesD = frames(dorsalIdx,:);
     majoraxisL = majoraxes(lateralIdx,:);
     majoraxisD = majoraxes(dorsalIdx,:);
     
     time = framesL .* 0.0025; % frames to seconds
     frames = framesL
     
     xL = centroidL(:,1); % gives 'intuitive' xy variables for easier plotting
     xD = centroidD(:,1);
     yL = centroidL(:,2);
     yD = centroidD(:,2);
     
     yL = (240 * calib) - yL; % position in mm coordinates
     yD = (240 * calib) - yD;
     
     
     % more conversion from px to mm and y-flipping
     heightLateralFloor = 240 - heightLateralBottom;
     obsL = obstacleLleft * calib;
     obsR = obstacleLright * calib;
     floorL = heightLateralFloor * calib;
     
     % calculates ground clearance and corrects for distance to mirror
     clearance = yL - floorL;
     bbD = single(bboxD);
     bbL = single(bboxL);
     scaleFactor = bbD(:,3) ./ bbL(:,3);
     clearanceCorr = clearance .* scaleFactor;

%release videos     
release(videoOrig);
release(videoFReader);


%%
% Calculates distance walked between 2 frames
xL = double(xL);
yL = double(yL);
xD = double(xD);
yD = double(yD);

distanceL = zeros(1,1);
distanceD = zeros(1,1);
velocityL = zeros(1,1);
velocityD = zeros(1,1);
frameDifferencesL = zeros(1,1);
frameDifferencesD = zeros(1,1);

binnedVelocitiesL = zeros(10,1);
binnedVelocitiesD = zeros(10,1);

% ERROR: velocity is computed for one row in the future, should be one
% frame back (see script structure_test)
row = 1;
while row < length(xL)
distL = sqrt((xL(row,1) - xL(row+1,1))^2+(yL(row,1) - yL(row+1,1))^2);
distanceL = [distanceL;distL];
frameDiffL = time(row+1,1) - time(row,1);
frameDifferencesL = [frameDifferencesL;frameDiffL];
row = row+1;
end

velocityL = distanceL./frameDifferencesL; % Velocity in mm/s

% takes walking average of 10 frames for smoothing
row = 10;
while row < length(velocityL)
    binVelL = (velocityL(row-9,1) + velocityL(row-8,1) +velocityL(row-7,1) +velocityL(row-6,1) + velocityL(row-5,1)...
        + velocityL(row-4,1) + velocityL(row-3,1) + velocityL(row-2,1) + velocityL(row-1,1) + velocityL(row,1))./10;
    binnedVelocitiesL = [binnedVelocitiesL;binVelL];
    row = row+1;
end

row = 1;
while row < length(xD)
distD = sqrt((xD(row,1) - xD(row+1,1))^2+(yD(row,1) - yD(row+1,1))^2);
distanceD = [distanceD;distD];
frameDiffD = time(row+1,1) - time(row,1);
frameDifferencesD = [frameDifferencesD;frameDiffD];
row = row+1;
end

velocityD = distanceD./frameDifferencesD; % Velocity in mm/s

% takes walking average of 10 frames for smoothing
row = 10;
while row < length(velocityD)
    binVelD = (velocityD(row-9,1) + velocityD(row-8,1) +velocityD(row-7,1) +velocityD(row-6,1) + velocityD(row-5,1)...
        + velocityD(row-4,1) + velocityD(row-3,1) + velocityD(row-2,1) + velocityD(row-1,1) + velocityD(row,1))./10;
    binnedVelocitiesD = [binnedVelocitiesD;binVelD];
    row = row+1;
end

% Corrects angle in lateral view
widthL = bsxfun(@times, majoraxisL, cos(orientationL));
heightL = bsxfun(@times, majoraxisL, sin(orientationL));
temp = 1./cos(orientationD);
widthLCorr = bsxfun(@times,widthL,temp);
majoraxisLCorr = sqrt(((widthLCorr).^2)+((heightL).^2));
orientationLcorr = asin(heightL./majoraxisLCorr);
degLcorr = radtodeg(orientationLcorr);

% cleanup temporary variables
clearvars -except filename trunkName videoList it scaleFactor calib clearanceCorr ...
    frameID centroidL centroidD ... 
    areasL areasD ...
    bboxL bboxD ...
    orientationL orientationD orientationLcorr ...
    frames ...
    objectsL objectsD ...
    majoraxisL majoraxisD ...
    heightLateralTop heightLateralBottom heightDorsalTop heightDorsalBottom  floorL...
    cropLeft cropRight obstacleLleft obstacleLright obstacleDleft obstacleDright obsL obsR...
    x y xL xD yL yD ...
    distanceL distanceD binnedVelocitiesL binnedVelocitiesD velocityL velocityD;

%%
% plot some results
results = figure;
subplot(3,2,1); plot(xL,orientationLcorr)
title('lateral orientation (rad, perspective corrected)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     
subplot(3,2,2); plot(xD,orientationD)
title('dorsal orientation (rad)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     
subplot(3,2,3); plot(xL,yL)
title('lateral position (centroid xy)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     
subplot(3,2,4); plot(xD,yD)
title('dorsal position (centroid xy)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     
subplot(3,2,5); plot(xL,clearanceCorr)
title('ground clearance (mm, perspective corrected)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     
subplot(3,2,6); plot(xD,binnedVelocitiesD)
title('Velocity (mm/s)');
ylim=get(gca,'ylim');
     line([obsL;obsL],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);
     line([obsR;obsR],ylim.',...
         'linewidth',1,...
         'color',[0,0,0]);

% Save prompt
qstring = sprintf('save results and figure as %s?',trunkName);
button = questdlg(qstring)
switch button
    case 'Yes'
        disp('saved!')
        save(trunkName);
        saveas(results,trunkName,'fig');
end

% Next file, repeat, end
qstring = sprintf('On to the next file?');
button = questdlg(qstring,'done','Next file','Repeat','End','Next file')
switch button
    case 'Next file'
        it = it+1;
    case 'Repeat'
        %it = it (stays the same)
    case 'End'
        it = length(videoList) + 1;
        close all;
end

end