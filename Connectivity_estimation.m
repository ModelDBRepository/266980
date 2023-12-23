%%%%%%% File for Connectivity estimation %%%%%%%
% 
% Time domain connectivity estimators: One connectivity matrix (nROI X nROI)
% 1) Correlation (non-directed measure)
% 2) Delayed Correlation (directed measure)
% 3) Temporal Granger Causality (directed measure)
% 4) Phase Synchronization (non-directed measure)
% 5) Transfer Entropy (directed measure)
% 
% Frequency domain connectivity estimators: One connectivity matrix (nROI X nROI) for each frequency band
% 1) Coherence (non-directed measure)
% 2) Lagged Coherence (non-directed measure)
% 3) Spectral Granger Causality (directed measure)

% In the Connectivity matrix the element (i,j) denotes the connectivity from signal X(:,j) to signal X(i,:) 
%(target ROIs in row, source ROIs in column)

clear
clc
close all

file_name = 'simulation';
eval(['load ' file_name]) % load data

nTrial=10;     %number of traials
nROI=4;        %number of ROIs

% Assign parameters for frequency domain analysis
Fs=100;        % Sampling rate
window=50;     
NFFT=1000;     
noverlap=[];
F=[0:Fs/(NFFT):Fs/2]; % frequency vector

% Define frequency bands: tot, theta, alpha, beta, gamma
i_min_tot=find(F==2);  % min tot frequency
i_max_tot=find(F==50); % min tot frequency
i_min_t=find(F==4);    % min theta frequency
i_max_t=find(F==8);    % max theta frequency
i_min_a=i_max_t+1;     % min alpha frequency
i_max_a=find(F==14);   % max alpha frequency
i_min_b=i_max_a+1;     % min beta frequency
i_max_b=find(F==25);   % max beta frequency
i_min_g=find(F==30);     % min gamma frequency
i_max_g=find(F==40);   % max gamma frequency


%% Correlation 
 
Corr_trial=zeros(nROI,nROI,nTrial);

for k = 1 : nTrial
    eeg = [Matrix_eeg_C(:,:,k)]';      
    Corr_trial(:,:,k)=corr(eeg);       
end
Connectivity_corr = mean(Corr_trial,3);

%% Delayed correlation

delayed_corr=zeros(nROI,nROI,nTrial);
delays = [0:5];
L_delays = length(delays);
array = zeros(nROI,nROI,L_delays);

for kd = 1:L_delays
    delay = delays(kd);
for trial = 1:nTrial
    eeg = Matrix_eeg_C(:,:,trial);
    for i=1:nROI
        for j=1:nROI
            if (i==j)
                delayed_corr(i,j,trial)=0;  
            else
                % delayed effect
                delayed_corr(i,j,trial) = corr(eeg(i,delay+1:end)',eeg(j,1:end-delay)');
            end
        end
    end
end
array(:,:,kd) = mean(delayed_corr,3);
end
    
[mm,indice] = max(abs(array),[],3);
Connectivity_delayed_corr=zeros(nROI,nROI);
for i=1:nROI
    for j=1:nROI
    Connectivity_delayed_corr(i,j)  = array(i,j,indice(i,j));
    end
end


%% Coherence 


Cxy = zeros(nROI,nROI,length(F));
for trial =1:nTrial
    eeg=Matrix_eeg_C(:,:,trial);
for pop1 = 1:nROI
    for pop2 = 1:nROI  
    Cxy(pop1,pop2,:) = squeeze(Cxy(pop1,pop2,:)) + mscohere(eeg(pop1,:),eeg(pop2,:),window,[],NFFT,Fs);
    end
end
end
Cxy_med=Cxy/nTrial;


% One Connectivity matrix for each frequency band
Connectivity_Coh_tot=mean(Cxy_med(:,:,i_min_tot:i_max_tot),3);
Connectivity_Coh_theta=mean(Cxy_med(:,:,i_min_t:i_max_t),3);
Connectivity_Coh_alpha=mean(Cxy_med(:,:,i_min_a:i_max_a),3);
Connectivity_Coh_beta=mean(Cxy_med(:,:,i_min_b:i_max_b),3);
Connectivity_Coh_gamma=mean(Cxy_med(:,:,i_min_g:i_max_g),3);


%% Lagged Coherence


CSPECTRUM=cell(nTrial,nROI); 
SPECTRUM=cell(nTrial,1);  


for is=1:nTrial
    
    eeg = Matrix_eeg_C(:,:,is)'; 
    for ir=1:nROI
       CSPECTRUM{is,ir}=cpsd(eeg(:,ir),eeg(:,1:ir),window,noverlap,NFFT,Fs);
    end  
    SPECTRUM{is,1}=pwelch(eeg,window,noverlap,NFFT,Fs);
    
end

COHERENCY=cell(nTrial,nROI);
LagCOH=cell(nTrial,nROI);

LagCoh_TOT=zeros(nROI,nROI,nTrial);
LagCoh_Theta=zeros(nROI,nROI,nTrial);
LagCoh_Alpha=zeros(nROI,nROI,nTrial);
LagCoh_Beta=zeros(nROI,nROI,nTrial);
LagCoh_Gamma=zeros(nROI,nROI,nTrial);


for is=1:nTrial

SPECTRUM_trial=SPECTRUM{is,1};

for ir=1:nROI

    CSPECTRUM_trial_ROI=CSPECTRUM{is,ir};

    for k=ir+1:nROI
        CSPECTRUM_trial_ROI(:,k)=conj(CSPECTRUM{is,k}(:,ir));
    end

    COHERENCY{is,ir}=(CSPECTRUM_trial_ROI)./...
        sqrt((repmat(SPECTRUM_trial(:,ir),1,nROI).*SPECTRUM_trial));

    LagCOH{is,ir}=((imag(COHERENCY{is,ir})).^2)./(1-((real(COHERENCY{is,ir})).^2));

    LagCoh_TOT(ir,:,is)=mean(LagCOH{is,ir}(i_min_tot:i_max_tot,:));
    LagCoh_Theta(ir,:,is)=mean(LagCOH{is,ir}(i_min_t:i_max_t,:));
    LagCoh_Alpha(ir,:,is)=mean(LagCOH{is,ir}(i_min_a:i_max_a,:)); 
    LagCoh_Beta(ir,:,is)=mean(LagCOH{is,ir}(i_min_b:i_max_b,:));
    LagCoh_Gamma(ir,:,is)=mean(LagCOH{is,ir}(i_min_g:i_max_g,:));


end
end

% One Connectivity matrix for each frequency band
Connectivity_LagCoh_TOT=mean(LagCoh_TOT,3);
Connectivity_LagCoh_Theta=mean(LagCoh_Alpha,3);
Connectivity_LagCoh_Alpha=mean(LagCoh_Theta,3);
Connectivity_LagCoh_Beta=mean(LagCoh_Beta,3);
Connectivity_LagCoh_Gamma=mean(LagCoh_Gamma,3);


%%  Temporal Granger Causality

inputs.nTrials=nTrial;
inputs.standardize=1;
inputs.flagFPE=false;
order=30;

[Connectivity_Granger_t, pvalue] = granger_time_connectivity(Data, order, inputs);

%% Spectral granger causality 

inputs.nTrials=nTrial;
inputs.freqResolution=0.1;
inputs.freq=0:0.1:Fs/2;
inputs.standardize=1; %0
inputs.flagFPE=false; %true
order=10;

[frequency_Grang, freq] = granger_spectral_connectivity(Data, Fs, order, inputs);
% frequency_Grang matrix has dimension nSignals x nSignals x nFreq
% the element (i,j,k) of the matrix indicates the connectivity from signal X(j,:) to signal X(i,:) at frequency f(k)


Connectivity_Granger_f_tot=mean(frequency_Grang(:,:,i_min_tot:i_max_tot),3,'omitnan');
Connectivity_Granger_f_theta=mean(frequency_Grang(:,:,i_min_t:i_max_t),3,'omitnan');
Connectivity_Granger_f_alpha=mean(frequency_Grang(:,:,i_min_a:i_max_a),3,'omitnan');
Connectivity_Granger_f_beta=mean(frequency_Grang(:,:,i_min_b:i_max_b),3,'omitnan');
Connectivity_Granger_f_gamma=mean(frequency_Grang(:,:,i_min_g:i_max_g),3,'omitnan');

%% phase synchronization

Con_trial = zeros(nROI,nROI,nTrial);

for trial = 1:nTrial
    
    eeg = Matrix_eeg_C(:,:,trial);
    eeg = eeg - mean(eeg,2);
    XX = hilbert(eeg');
    Phase = angle(XX);
    Con = zeros(nROI,nROI);
    
    for k = 1:nROI
        for h = 1:nROI
            if ne(k,h)
                phase_diff = Phase(:,h) - Phase(:,k);
                media = mean(exp(1i*phase_diff));
                Con(h,k) = abs(media);
            end
        end
    end
    Con_trial(:,:,trial) = Con;
end

Connectivity_phaseSync = mean(Con_trial,3);

%% Transfer entropy (TRENTOOL needed)

% Add path to TRENTOOL folder
oldpath = addpath ('..\TRENTOOL3') ;

ft_defaults;
OutputDataPath='TE_output';

% Define input data to TRENTOOL
data=[];
data.fsample=Fs; %samplig rate

% Assign a time vector to each trial
data.time=cell(nTrial,1);
for i=1:nTrial
    data.time{i,1}=tt;
end

% Assign ROI labels
data.label=cell(nROI,1);
data.label{1,1}='ROI 1'; % ROI beta
data.label{2,1}='ROI 2'; % ROI gamma
data.label{3,1}='ROI 3'; % ROI theta
data.label{4,1}='ROI 4'; % ROI alpha

% Assign data for each trial (each trial holds samples from all ROIs)
data.trial=cell(1,nTrial);
for j=1:nTrial
   data.trial{1,j}=Matrix_eeg_C(:,:,j);
end

% Define cfg for TEprepare
cfgTEP=[];
cfgTEP.toi=[min(data.time{1,1}),max(data.time{1,1})];
cfgTEP.channel=data.label;
cfgTEP.predicttime_u=10;
cfgTEP.predicttimemax_u=11;
cfgTEP.predicttimemin_u=9;
cfgTEP.predicttimestepsize=1;

cfgTEP.TEcalctype='VW_ds';
cfgTEP.trialselect='no';
cfgTEP.minnrtrials=1;
cfgTEP.actthrvalue=30;

cfgTEP.optimizemethod='ragwitz';
cfgTEP.ragdim=4:16;
cfgTEP.ragtaurange=[0.5 1];
cfgTEP.ragtausteps=10;
cfgTEP.flagNei='Mass';   
cfgTEP.sizeNei=4;
cfgTEP.repPred=100;

data_prepared=TEprepare(cfgTEP,data);

% Define cfg for TEsurrogatestats or InteractionDelayReconstruction_calculate
cfgTESS=[];
cfgTESS.optdimusage='indivdim';
cfgTESS.tail=1;
cfgTESS.surrogatetype='trialshuffling';
cfgTESS.extracond='Faes_Method';
cfgTESS.shifttest='no';
cfgTESS.MIcalc=1;
cfgTESS.fileidout=strcat(OutputDataPath);

TGA_results=InteractionDelayReconstruction_calculate(cfgTEP,cfgTESS,data);

% Graph correction for multivariate effects
% Define cfg for TEgraphanalysis
cfgGA=[];
cfgGA.threshold=2;
cfgGA.cmc=1;
TGA_results_GA=TEgraphanalysis(cfgGA,TGA_results);
% save([OutputDataPath '_results.mat'],'TGA_results_GA');


% Extracting TE matrix
PermMat=TGA_results_GA.TEpermvalues;
Comb_ord=TGA_results_GA.sgncmb;
vMat=PermMat(:,4); % selecting the fourth column of PermMat containing TE values sorted accrding to Comb_ord

Matrix=zeros(nROI,nROI);
ind_v=1:nROI-1;

for i=1:nROI
    index=1:nROI;
    index(i)=[];
    Matrix(index,i)=vMat(ind_v,1);
    ind_v=ind_v+(nROI-1);
end
    
Connectivity_TE=Matrix;
