function [] = lc_resultsAnalysis(typeMarkers, iXp, branchNames, statEvalType)

load('mat/allPath.mat');

lType = { 'rx-.', 'gx-.',  'bx-.', 'mx-.' , 'rs-', 'gs-',  'bs-', 'ms-' };

nType = 4;

% Legend properties: set Title, axis size
tSize = 15;
% Set Legend size
lSize = 8;%20
% Set FontUnit size
uSize = 15;
% Set line width
lWidth = 3;
% Set marker size
mSize = 10;
%%%

displayDetectRate = 1;
displayDetectPlusIdentRate = 1;
displayAccuracy = 1;
nSubPlot = displayDetectRate + displayDetectPlusIdentRate + displayAccuracy;


%hFig = figure(1);
%set(hFig, 'Position', [100 100 1400 500]);
h1 = figure('Position',[50 50 400 400]);
h2 = figure('Position',[500 50 400 400]);
h3 = figure('Position',[950 50 400 400]);

if ( length(branchNames) > 1 ) && ( length(typeMarkers) > 1 )
    error('branchNames or typeMarkers must be of length 1');
end

if ( iXp == 1 )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      Results vs. distance                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iTextLegDetection = 1;
    iTextLegConfusion = 1;
    iTextLegPrecision = 1;
    
    for iBranch = 1:length(branchNames)
        for iType = 1:length(typeMarkers)
            
            fn = allResultPath{iType, iBranch};
            resultData = load( fn.statisticalEvalResultPath );
            
            if length(typeMarkers) > 1
                iLegend = iType;
                textLeg = typeMarkers{iType};
            else
                iLegend = iBranch;
                textLeg = branchNames{iBranch};
            end
            
            iSubPlot = 1;
            nCrowns = typeMarkers{iType};
            
            resultData.nbNegatifs = 1 - resultData.nbNegatifs/resultData.nTest;
            
            if displayDetectRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h1); hold on;
                iSubPlot = iSubPlot+1;
                plot(resultData.distance,resultData.nbNegatifs(:,1), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbNegatifs(:,2), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbNegatifs(:,3), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbNegatifs(:,4), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - \theta = 0$' ];  iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - \theta = 25$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - \theta = 50$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - \theta = 75$' ]; iTextLegDetection = iTextLegDetection + 1;
                
                L = legend( textLegendDetection{:} ,'Location','SouthWest');
                
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Distance $D$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_n$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                set(gca,'FontSize',uSize);
                
                axis([ min(resultData.distance) max(resultData.distance)  0 1.5 ]);
                
                % Save figures
                %saveas(h1,'figure/xp2.pdf','pdf');
                %saveas(h1,'figure/xp2.fig','fig');
                
            end
            
            resultData.nbConfusion = 1 - resultData.nbConfusion/resultData.nTest;
            
            if displayDetectPlusIdentRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h2); hold on;
                iSubPlot = iSubPlot+1;
                
                plot(resultData.distance,resultData.nbConfusion(:,1), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbConfusion(:,2), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbConfusion(:,3), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,resultData.nbConfusion(:,4), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                %%%
                set(gca,'FontSize',uSize);
                
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - \theta = 0$' ];  iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - \theta = 25$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - \theta = 50$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - \theta = 75$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                
                L = legend( textLegendConfusion{:} ,'Location','SouthWest');
                
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Distance $D$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_c$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                set(gca,'FontSize',uSize);
                %saveas(h2,'figure/xp2-c.pdf','pdf');
                
                axis([ min(resultData.distance) max(resultData.distance)  0 1.5 ]);
                
            end
            
            if displayAccuracy
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h3); hold on;
                iSubPlot = iSubPlot+1;

                if strcmp(statEvalType, 'mean')
                    precision = mean( resultData.precision(:,:,:) , 2);
                elseif strcmp(statEvalType, 'median')
                    precision = median( resultData.precision(:,:,:) , 2);
                else
                    error('Statistical criterion unknown');
                end
                
                plot(resultData.distance,precision(:,1), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,precision(:,2), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,precision(:,3), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.distance,precision(:,4), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                %plot(lengthMotionBlur,nbConfusion(5,:), 'kd-','LineWidth',3,'MarkerEdgeColor','k','MarkerSize',10);
                %axis([0 10 0.4 1])
                %%%
                set(gca,'FontSize',uSize);
                
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 0$' ];  iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 25$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 50$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 75$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                
                L = legend( textLegendPrecision{:} ,'Location','NorthWest');
                
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Distance $D$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('Imaged center error (pixels)','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                
                axis([ min(resultData.distance) max(resultData.distance)  0 2 ]);
                
            end
        end
    end
elseif ( iXp == 2 )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   Results vs. motion blur                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iTextLegDetection = 1;
    iTextLegConfusion = 1;
    iTextLegPrecision = 1;
    
    for iBranch = 1:length(branchNames)
        for iType = 1:length(typeMarkers)
            
            fn = allResultPath{iType, iBranch};
            resultData = load( fn.statisticalEvalResultPath );
            
            if length(typeMarkers) > 1
                iLegend = iType;
                textLeg = typeMarkers{iType};
            else
                iLegend = iBranch;
                textLeg = branchNames{iBranch};
            end
            
            iSubPlot = 1;
            
            nCrowns = typeMarkers{iType};
            
            %resultData = load([ 'mat/' branchNames{iBranch} '/' nCrowns '/result/' int2str(iXp) '.mat' ]);
            
            resultData.nbNegatifs = 1 - resultData.nbNegatifs/resultData.nTest;
            
            if displayDetectRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h1); hold on;
                iSubPlot = iSubPlot+1;
                
                %h1 = figure(1); hold on;
                plot(resultData.lengthMotionBlur,resultData.nbNegatifs(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbNegatifs(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbNegatifs(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbNegatifs(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                
                %%%
                
                textLegendDetect{iTextLegDetection} = [ '$' textLeg ' - D = 10$' ];  iTextLegDetection = iTextLegDetection + 1;
                textLegendDetect{iTextLegDetection} = [ '$' textLeg ' - D = 20$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetect{iTextLegDetection} = [ '$' textLeg ' - D = 30$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetect{iTextLegDetection} = [ '$' textLeg ' - D = 40$' ]; iTextLegDetection = iTextLegDetection + 1;
                
                L = legend( textLegendDetect{:} ,'Location','SouthWest');
                
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Longueur $l (pixels)$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_n$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                set(gca,'FontSize',uSize);
                
                axis([min(resultData.lengthMotionBlur) max(resultData.lengthMotionBlur) 0 1.5 ]);
                
                % Save figures
                %saveas(h1,'figure/xp2.pdf','pdf');
                %saveas(h1,'figure/xp2.fig','fig');
                %%%
                
            end
            
            resultData.nbConfusion = 1 - resultData.nbConfusion/resultData.nTest;
            
            if displayDetectPlusIdentRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h2); hold on;
                iSubPlot = iSubPlot+1;
                
                %h2 = figure(2); hold on;
                plot(resultData.lengthMotionBlur,resultData.nbConfusion(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbConfusion(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbConfusion(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,resultData.nbConfusion(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                axis([0 10 0.4 1])
                %%%
                set(gca,'FontSize',uSize);
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 10$' ];  iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 20$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 30$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 40$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                
                L = legend( textLegendConfusion{:} ,'Location','SouthWest');
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Length $l (pixels)$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_c$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                %saveas(h2,'figure/xp2-c.pdf','pdf');
                %saveas(h2,'figure/xp2-c.fig','fig');
                %%%
                
                axis([min(resultData.lengthMotionBlur) max(resultData.lengthMotionBlur) 0 1.5 ]);
                
            end
            
            if displayAccuracy
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h3); hold on;
                iSubPlot = iSubPlot+1;
                
                if strcmp(statEvalType, 'mean')
                    precision = mean( resultData.precision(:,:,:) , 2);
                elseif strcmp(statEvalType, 'median')
                    precision = median( resultData.precision(:,:,:) , 2);
                else
                    error('Statistical criterion unknown');
                end

                %h3 = figure(3); hold on;
                plot(resultData.lengthMotionBlur,precision(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,precision(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,precision(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.lengthMotionBlur,precision(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                %plot(lengthMotionBlur,nbConfusion(5,:), 'kd-','LineWidth',3,'MarkerEdgeColor','k','MarkerSize',10);
                %axis([0 10 0.4 1])
                %%%
                set(gca,'FontSize',uSize);
                
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - D = 10$' ];  iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - D = 20$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - D = 30$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - D = 40$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                
                L = legend( textLegendPrecision{:} ,'Location','NorthWest');
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Distance $D$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('Imaged center error (pixels)','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                
            end
        end
    end
elseif ( iXp == 3 )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                Results vs. the % of occlusion                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iTextLegDetection = 1;
    iTextLegConfusion = 1;
    iTextLegPrecision = 1;
    
    nSubPlot = displayDetectRate + displayDetectPlusIdentRate + displayAccuracy;
    
    for iBranch = 1:length(branchNames)
        for iType = 1:length(typeMarkers)
            
            fn = allResultPath{iType, iBranch};
            resultData = load( fn.statisticalEvalResultPath );
            
            if length(typeMarkers) > 1
                iLegend = iType;
                textLeg = typeMarkers{iType};
            else
                iLegend = iBranch;
                textLeg = branchNames{iBranch};
            end
            
            iSubPlot = 1;
            
            nCrowns = typeMarkers{iType};
            
            resultData.nbNegatifs = 1 - resultData.nbNegatifs/resultData.nTest;
            
            % Detection rate
            if displayDetectRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h1); hold on;
                iSubPlot = iSubPlot+1;

                plot(resultData.occlusion,resultData.nbNegatifs(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbNegatifs(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbNegatifs(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbNegatifs(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                
                %%%
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - D = 10$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - D = 20$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - D = 30$' ]; iTextLegDetection = iTextLegDetection + 1;
                textLegendDetection{iTextLegDetection} = [ '$' textLeg ' - D = 40$' ]; iTextLegDetection = iTextLegDetection + 1;
                
                L = legend( textLegendDetection{:} ,'Location','SouthWest');
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Occlusion $(\%)$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_n$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                set(gca,'FontSize',uSize);
                
                axis([min(resultData.occlusion) max(resultData.occlusion) 0 1.5 ]);
                
                % Save figures
                %saveas(h1,'figure/xp2.pdf','pdf');
                %saveas(h1,'figure/xp2.fig','fig');
                %%%
                
            end
            
            resultData.nbConfusion = 1 - resultData.nbConfusion/resultData.nTest;
            
            % Identification rate
            if displayDetectPlusIdentRate
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h2); hold on;
                iSubPlot = iSubPlot+1;
                %h2 = figure(2); hold on;
                
                plot(resultData.occlusion,resultData.nbConfusion(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbConfusion(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbConfusion(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion,resultData.nbConfusion(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                %%%
                set(gca,'FontSize',uSize);
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 10$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 20$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 30$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                textLegendConfusion{iTextLegConfusion} = [ '$' textLeg ' - D = 40$' ]; iTextLegConfusion = iTextLegConfusion + 1;
                
                L = legend( textLegendConfusion{:} ,'Location','SouthWest');
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Occlusion $(\%)$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('$1$ - $\tau_c$','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                set(gca,'FontSize',uSize);
                %saveas(h2,'figure/xp2-c.pdf','pdf');
                %saveas(h2,'figure/xp2-c.fig','fig');
                %%%
                
                axis([min(resultData.occlusion) max(resultData.occlusion) 0 1.5 ]);
                
            end
            
            % Precision of the imaged center estimation
            if displayAccuracy
                
                %subplot(1,nSubPlot,iSubPlot); hold on;
                figure(h3); hold on;
                iSubPlot = iSubPlot+1;

                if strcmp(statEvalType, 'mean')
                    precision = mean( resultData.precision(:,:,:) , 2);
                elseif strcmp(statEvalType, 'median')
                    precision = median( resultData.precision(:,:,:) , 2);
                else
                    error('Statistical criterion unknown');
                end

                %h3 = figure(3); hold on;
                plot(resultData.occlusion, precision(1,:), lType{nType*(iLegend-1)+1},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion, precision(2,:), lType{nType*(iLegend-1)+2},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion, precision(3,:), lType{nType*(iLegend-1)+3},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                plot(resultData.occlusion, precision(4,:), lType{nType*(iLegend-1)+4},'LineWidth',lWidth,'MarkerEdgeColor','k','MarkerSize',mSize);
                %plot(lengthMotionBlur,nbConfusion(5,:), 'kd-','LineWidth',3,'MarkerEdgeColor','k','MarkerSize',10);
                %axis([0 10 0.4 1])
                %%%
                set(gca,'FontSize',uSize);
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 0$' ];  iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 25$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 50$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                textLegendPrecision{iTextLegPrecision} = [ '$' textLeg ' - \theta = 75$' ]; iTextLegPrecision = iTextLegPrecision + 1;
                
                L = legend( textLegendPrecision{:} ,'Location','NorthWest');
                set(L,'Interpreter','Latex');
                set(L,'FontSize',lSize);
                
                X = xlabel('Occlusion $(\%)$','FontSize',tSize);
                set(X,'Interpreter','Latex');
                Y = ylabel('Imaged center error (pixels)','FontSize',tSize);
                set(Y,'Interpreter','Latex');
                
                axis([min(resultData.occlusion) max(resultData.occlusion) 0 1.5 ]);
                
            end
        end
    end
end
