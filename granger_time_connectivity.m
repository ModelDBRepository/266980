function [connectivity, pValue] = granger_time_connectivity(X, order, inputs)
% BST_GRANGER       Granger causality in mean (regular log-GC from Geweke1982) 
%                   modificato da BST_GRANGER (ho tolto la Granger
%                   Causality in variance: information statistic from
%                   Hafner2007)
%                   
%
% Inputs:
%   X         - set of signals, one signal per row
%                   [X: MX x N  MX= number of signals, N=overall number of samples]
%   order         - maximum lag in AR model for causality in mean
%                   [p: nonnegative integer]
%   inputs        - structure of parameters:
%   |-nTrials     - # of trials in concantenated signal
%   |               [T: positive integer]
%   |-standardize - if true (default), remove mean from each signal.
%   |               if false, assume signal has already been detrended
%   |-flagFPE     - if true, optimize order for AR model
%   |               if false (default), force same order in all AR models
%   |               [E: default false]
%
% Outputs:
%   connectivity  - A x B matrix of causalities in mean from source to sink
%                   [C: MX x MY(=MX) matrix]
%   pValue        - parametric p-value for corresponding Granger causality in
%                   mean estimate
%                   [P: MX x MY(=MX) matrix]
% See also BST_MVAR, BST_VGARCH.
%
% For each signal pair (a,b) we calculate the Granger causality in mean GC(a,b):
%                        Var(x_a[t] | x_a[t-1, ..., t-k])         
%              ----------------------------------------------------
%              Var(x_a[t] | x_a[t-1, ..., t-k], y_b[t-1, ..., t-k])
% If Y is empty or Y = X, we set element GC(a,a) to be zero.
% 
% 
% Call:
%   [connectivity, p] = bst_granger(X, 20, inputs); 
%   inputs.nTrials = 10; % use trial-averaged covariances in AR estimation
%   inputs.standardize = true; % zero mean and unit variance
%   inputs.flagFPE = true; % allow different orders for each pair of signals

% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2020 University of Southern California & McGill University
% This software is distributed under the terms of the GNU General Public License
% as published by the Free Software Foundation. Further details on the GPLv3
% license can be found at http://www.gnu.org/copyleft/gpl.html.
% 
% FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE
% UNIVERSITY OF SOUTHERN CALIFORNIA AND ITS COLLABORATORS DO NOT MAKE ANY
% WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY
% LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
%
% For more information type "brainstorm license" at command prompt.
% =============================================================================@
%
% Authors: Sergul Aydore & Syed Ashrafulla, 2012

% default: 1 trial
if ~isfield(inputs, 'nTrials') || isempty(inputs.nTrials)
  inputs.nTrials = 1;
end

% default: do not optimize order in MVAR modeling
if ~isfield(inputs, 'flagFPE') || isempty(inputs.flagFPE)
  inputs.flagFPE = false;
end

nX = size(X, 1);
nSamples = size(X, 2);
nTimes = nSamples / inputs.nTrials;


% remove linear effects if desired
if isfield(inputs, 'standardize') && inputs.standardize
  detrender = [ ...
    ones(1, nTimes); ... % constant trend in data
    linspace(-1, 1, nTimes); ... % linear trend in data
    3/2 * linspace(-1, 1, nTimes).^2 - 1/2 ... % quadratic trend in data
  ];
  
  % detrend X
  for iTrial = 1:inputs.nTrials
    X(:, (iTrial-1)*nTimes + (1:nTimes)) = X(:, (iTrial-1)*nTimes + (1:nTimes)) - ( X(:, (iTrial-1)*nTimes + (1:nTimes)) / detrender ) * detrender;
    X(:, (iTrial-1)*nTimes + (1:nTimes)) = diag( sqrt(sum(X(:, (iTrial-1)*nTimes + (1:nTimes)).^2, 2)) ) \ X(:, (iTrial-1)*nTimes + (1:nTimes));
  end
  
end

%% Iterate over all pairs of sinks & sources

% for causality in mean we need the restricted variance
restOrder = zeros(nX, 1); restCovFull = zeros(nX, order+1);
for iX = 1:nX
  [syed, syed, restOrder(iX), syed, syed, restCovFull(iX, :)] = bst_mvar(X(iX, :), order, inputs.nTrials, inputs.flagFPE); %#ok<ASGLU>
end
  
  % setup
  connectivity = zeros(nX);
  
  % only iterate over one triangle
  for iX = 1:nX
    
    % iterate over all the pairs after iX
    for iY = (iX+1):nX
      
      % bivariate autoregressive model with sink_a and sink_b
      [syed, syed, unOrder, syed, syed, unCovFull, residual] = bst_mvar([X(iX, :); X(iY, :)], order, inputs.nTrials, inputs.flagFPE); %#ok<ASGLU>
      
      % causality in mean: Geweke-Granger, i.e. restricted variance / unrestricted variance - 1
      if inputs.flagFPE % get the minimum order of the two models estimated
        
        % source = iY, sink = iX
        minOrder = min([restOrder(iX) unOrder]); 
        connectivity(iX, iY) = restCovFull(iX, minOrder+1) / unCovFull(1, 1, minOrder+1) - 1;
        
        % source = iX, sink = iY
        minOrder = min([restOrder(iY) unOrder]);
        connectivity(iY, iX) = restCovFull(iY, minOrder+1) / unCovFull(2, 2, minOrder+1) - 1;
        
      else % by default, bst_mvar sends the result of the single model of given order into the "Full" variables
        
        connectivity(iX, iY) = restCovFull(iX) / unCovFull(1, 1) - 1; % source = iY, sink = iX
        connectivity(iY, iX) = restCovFull(iY) / unCovFull(2, 2) - 1; % source = iX, sink = iY
        
      end
           
    end
    
    % diagonal will equal the maximum of all inflows and outflows for iX
    %connectivity(iX, iX) = max([connectivity(iX, :) connectivity(:, iX)']);
    
  end
  

%% Statistics: parametric p-values for causality in mean (based on regression coefficients) and variance (based on Wald statistics)

% causality in mean: F statistic of connectivity when multiplied by number of regressors
pValue = 1 - betainc(connectivity ./ (1 + connectivity), order / 2, (nSamples - order * inputs.nTrials - 2 * order - 1) / 2, 'lower');
% here we assume we have many more samples than the order of the MVAR model so that in all cases we use the second condition below to compute the p-value
% note: if connectivity = 0 (auto-causality or two of the same signals) then this formula evalutes pValue = 1 which is desired (no significant causality)
  
% causality in mean: F statistic of connectivity when multiplied by number of regressors
% tic
% iFlip = nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1 > connectivity * order;
% pValue = zeros(size(connectivity));
% pValue(~iFlip) = 1 - betainc(1 ./ (1 + connectivity(~iFlip)), (nSamples - order(~iFlip) * inputs.nTrials - 2 * order(~iFlip) - 1) / 2, order(~iFlip) / 2, 'upper');
% pValue(iFlip) = 1 - betainc(connectivity(iFlip) ./ (1 + connectivity(iFlip)), order(iFlip) / 2, (nSamples - order(iFlip) * inputs.nTrials - 2 * order(iFlip) - 1) / 2, 'lower');
% toc
% % fcdf(connectivity .* (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ order, order, nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1)
% % which for nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1 <= connectivity * order is
% % = betainc((nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1 + connectivity * (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / order * order), (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, order/2, 'upper')
% % = betainc((nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ ((nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) .* (1 + connectivity)), (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, order/2, 'upper')
% % = betainc(1 ./ (1 + connectivity), (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, order/2, 'upper')
% % and for nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1 >= connectivity * order is
% % = betainc(connectivity .* (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ order .* order ./ (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1 + connectivity .* (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ order .* order), order/2, (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, 'lower')
% % = betainc(connectivity .* (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) ./ ((nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) .* (1 + connectivity)), order/2, (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, 'lower')
% % = betainc(connectivity ./ (1 + connectivity), order/2, (nSamples - order(~iFlip) * inputs.nTrials - 2 * order - 1) / 2, 'lower')

% causality in mean: chi-square statistic of connectivity when multiplied by the number of samples
% tic
% pValue = 1 - gammainc(connectivity * (nSamples - order * inputs.nTrials) / 2, order / 2);
% toc
% % chi2cdf(connectivity * (nSamples - order * inputs.nTrials), order)
% % = gamcdf(connectivity * (nSamples - order * inputs.nTrials), order/2, 2)
% % = gammainc(connectivity * (nSamples - order * inputs.nTrials) / 2, order / 2)

% causality in variance: chi-square statistic of connectivity when multiplied by number of samples (minus lag minus 1) to get chi-square statistic
% pValueV = 1 - gammainc(connectivityV .* (nSamples - order * inputs.nTrials - inputs.lag - 1) / 2, inputs.lag);
% chi2cdf(connectivityV .* (nSamples - order * inputs.nTrials - inputs.lag - 1), 2 * inputs.lag)
% = gamcdf(connectivityV .* (nSamples - order * inputs.nTrials - inputs.lag - 1), (2 * inputs.lag)/2 = inputs.lag, 2)
% = gammainc(connectivityV .* (nSamples - order * inputs.nTrials - inputs.lag - 1) / 2, inputs.lag)
% note: if connectivityV = 0 (auto-causality or two of the same signals) then this formula evalutes pValueV = 1 which is desired (no significant causality)
