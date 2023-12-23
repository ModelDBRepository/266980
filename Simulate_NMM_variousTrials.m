% This program simulates 10 trials of 4 interconnected ROIs
clear
close all
clc

window = 50;  
zeropadding = 1000; 

Npop = 4; % Number of ROIs
dt=0.0001;
f_eulero = 1/dt;
tend = 1 + 10;
t=(0:dt:tend);
N=length(t);



%% parameters definition

e0 = 2.5; % Saturation value of the sigmoid
r = 0.56;  % Slope of the sigmoid(1/mV) 
s0 = 10;  % Center of the sigmoid
flag = 0;  % Parameter to set the working point in the central position

D=0.01*ones(1,Npop); % Delay between regions (10 ms)
G=[5.17 4.45 57.1];  % Synaptic gains


% ROI 1: Beta Rhythm
% Connectivity constants
C(1,1) = 54.; %Cep
C(1,2) = 54.; %Cpe
C(1,3) = 54.; %Csp
C(1,4) = 67.5; %Cps  
C(1,5) = 27.; %Cfs
C(1,6) = 54.; %Cfp
C(1,7) = 540.; %Cpf
C(1,8) = 10.; %Cff
a(1,:)=[68.5 30 300]; % Reciprocal of synaptic time constants (omega)       

        
% ROI 2: Gamma Rhythm
% Connectivity constants
C(2,1) = 54.; %Cep
C(2,2) = 54.; %Cpe
C(2,3) = 54.; %Csp
C(2,4) = 67.5; %Cps   
C(2,5) = 27.; %Cfs
C(2,6) = 108.; %Cfp
C(2,7) = 300.; %Cpf
C(2,8) = 10.; %Cff
a(2,:)=[125 30 400]; % Reciprocal of synaptic time constants (omega)  

        
% ROI 3: Theta Rhythm 
% Connectivity constants
C(3,1) = 54.; %Cep
C(3,2) = 54.; %Cpe
C(3,3) = 54.; %Csp
C(3,4) = 67.5; %Cps   
C(3,5) = 15.; %Cfs
C(3,6) = 27.; %Cfp
C(3,7) = 300.; %Cpf
C(3,8) = 10.; %Cff
a(3,:)=[75 30 300]; % Reciprocal of synaptic time constants (omega)  
        

% ROI 4: Alpha Rhythm
% Connectivity constants
C(4,1) = 54.; %Cep
C(4,2) = 54.; %Cpe
C(4,3) = 54.; %Csp
C(4,4) = 450; %Cps   
C(4,5) = 10.; %Cfs
C(4,6) = 35.; %Cfp
C(4,7) = 300.; %Cpf   
C(4,8) = 25.; %Cff
a(4,:)=[66 42 300]; % Reciprocal of synaptic time constants (omega)  

%% definition of excitatory and inhibitory synapses between ROIs
Wf=zeros(Npop); %inhibitory synapses
Wp=zeros(Npop); %excitatory synapses


% Synaptic configuration of the network in Figure 2A
Wp = [0 10 0 0   
      40 0 20 0   
      0 10 0 0   
      0 0 0 0]; 
  
Wf = [0 0 0 15  
      0 0 0 20  
      0 0 0 20   
      0 0 0 0]; 

%%

start = 10000; 
step_red = 100;   % step reduction from 10000 to 100 Hz
fs = f_eulero/step_red;


%% simulo vari trial
Ntrial = 10;
Matrix_eeg_C = zeros(Npop,(N-1-start)/step_red,Ntrial);  % exclusion of the first second due to a possible transitory
tt=zeros(1,(N-1-start)/step_red);
for trial = 1: Ntrial
    
% defining equations of a single ROI
yp=zeros(Npop,N);
xp=zeros(Npop,N);
vp=zeros(Npop,1);
zp=zeros(Npop,N);
ye=zeros(Npop,N);
xe=zeros(Npop,N);
ve=zeros(Npop,1);
ze=zeros(Npop,N);
ys=zeros(Npop,N);
xs=zeros(Npop,N);
vs=zeros(Npop,1);
zs=zeros(Npop,N);
yf=zeros(Npop,N);
xf=zeros(Npop,N);
zf=zeros(Npop,N);
vf=zeros(Npop,1);
xl=zeros(Npop,N);
yl=zeros(Npop,N);

% mean value of the input noise to each ROI (through excitatory interneurons)
m = zeros(Npop,1);
m(1) = 400;  % mean value of the Input Noise to ROI 1 (Beta)
m(2) = 400;  % mean value of the Input Noise to ROI 2 (Gamma)
m(3) = 400;  % mean value of the Input Noise to ROI 3 (Theta)
m(4) = 200;  % mean value of the Input Noise to ROI 4 (Alpha)
kmax=round(max(D)/dt);

% different seed for noise generation at each trial
rng(10+trial)  
sigma_p = sqrt(5/dt); % Standard deviation of the input noise to excitatory neurons
sigma_f = sqrt(5/dt); % Standard deviation of the input noise to inhibitory neurons
np = randn(Npop,N)*sigma_p; % Generation of the input noise to excitatory neurons
nf = randn(Npop,N)*sigma_f; % Generation of the input noise to inhibitory neurons

for k=1:N-1
   up=np(:,k)+m; % input of exogenous contributions to excitatory neurons
   uf=nf(:,k);  % input of exogenous contributions to inhibitory neurons
    
    if(k>kmax)
        for i=1:Npop
            up(i)=up(i)+Wp(i,:)*zp(:,round(k-D(i)/dt));
            uf(i)=uf(i)+Wf(i,:)*zp(:,round(k-D(i)/dt));
        end
    end
   
    % post-synaptic membrane potentials
    vp(:)=C(:,2).*ye(:,k)-C(:,4).*ys(:,k)-C(:,7).*yf(:,k);
    ve(:)=C(:,1).*yp(:,k);
    vs(:)=C(:,3).*yp(:,k);
    vf(:)=C(:,6).*yp(:,k)-C(:,5).*ys(:,k)-C(:,8).*yf(:,k)+yl(:,k);
    
    % average spike density
    zp(:,k)=2*e0./(1+exp(-r*(vp(:)-s0)))-flag*e0;
    ze(:,k)=2*e0./(1+exp(-r*(ve(:)-s0)))-flag*e0;
    zs(:,k)=2*e0./(1+exp(-r*(vs(:)-s0)))-flag*e0;
    zf(:,k)=2*e0./(1+exp(-r*(vf(:)-s0)))-flag*e0;
    
    
    % post synaptic potential for pyramidal neurons
    xp(:,k+1)=xp(:,k)+(G(1)*a(:,1).*zp(:,k)-2*a(:,1).*xp(:,k)-a(:,1).*a(:,1).*yp(:,k))*dt;  
    yp(:,k+1)=yp(:,k)+xp(:,k)*dt; 
    
    % post synaptic potential for excitatory interneurons
    xe(:,k+1)=xe(:,k)+(G(1)*a(:,1).*(ze(:,k)+up(:)./C(:,2))-2*a(:,1).*xe(:,k)-a(:,1).*a(:,1).*ye(:,k))*dt;  
    ye(:,k+1)=ye(:,k)+xe(:,k)*dt; 
    
    % post synaptic potential for slow inhibitory interneurons
    xs(:,k+1)=xs(:,k)+(G(2)*a(:,2).*zs(:,k)-2*a(:,2).*xs(:,k)-a(:,2).*a(:,2).*ys(:,k))*dt;   
    ys(:,k+1)=ys(:,k)+xs(:,k)*dt; 
    
    % post synaptic potential for fast inhibitory interneurons
    xl(:,k+1)=xl(:,k)+(G(1)*a(:,1).*uf(:)-2*a(:,1).*xl(:,k)-a(:,1).*a(:,1).*yl(:,k))*dt;  
    yl(:,k+1)=yl(:,k)+xl(:,k)*dt; 
    xf(:,k+1)=xf(:,k)+(G(3)*a(:,3).*zf(:,k)-2*a(:,3).*xf(:,k)-a(:,3).*a(:,3).*yf(:,k))*dt;   
    yf(:,k+1)=yf(:,k)+xf(:,k)*dt; 

end

% low pass filter at 50 Hz before resampling at 100 Hz
Omp = 50/(f_eulero/2);
Oms = 60/(f_eulero/2);
Rp = 1;
Rs = 40;
[Nfilter, Omn] = ellipord(Omp, Oms, Rp, Rs);
[B,A] = ellip(Nfilter,Rp,Rs,Omn);


eeg_tot=diag(C(:,2))*ye-diag(C(:,4))*ys-diag(C(:,7))*yf;
for j = 1:Npop
eeg_tot(j,:) = filtfilt(B,A,eeg_tot(j,:));
end
eeg = eeg_tot(:,start:step_red:end);
t_res=t(start:step_red:end);
% Matrix extraction for corr, delayed corr, coh, lagged coh, phase sync and TE estimation
Matrix_eeg_C(:,:,trial) = eeg(:,1:end-1); % matrix dimension= n°ROI x Nsamples x nTrials
end
tt=t_res(1:end-1);

% Matrix extraction for temporal and spectral Granger Causality estimation
Data = []; % trails concatenated in column
for j = 1:Ntrial
Data = [Data Matrix_eeg_C(:,:,j)]; % matrix dimension= n°ROI x (Nsamples x nTrials)
end

save simulation Matrix_eeg_C Data Wp Wf m tt

%% section that plots the temporal pattern of pyramidal potenital, pyramidal spike density, and the relative spectra
linea = 1;
fonte = 14;

passoin = 90000;
passofin = 99999;

figure(1)
subplot(2,2,1)
plot(t(passoin:passofin),zp(3,passoin:passofin),'k','linewidth',linea)
title('ROI\theta')
set(gca,'fontsize',fonte)
%xlabel('time (s)')
ylabel('spike density')
subplot(222)
plot(t(passoin:passofin),zp(4,passoin:passofin),'k','linewidth',linea)
title('ROI\alpha')
set(gca,'fontsize',fonte)
%xlabel('time (s)')
subplot(223)
plot(t(passoin:passofin),zp(1,passoin:passofin),'k','linewidth',linea)
title('ROI\beta')
set(gca,'fontsize',fonte)
xlabel('time (s)')
ylabel('spike density')
subplot(224)
plot(t(passoin:passofin),zp(2,passoin:passofin),'k','linewidth',linea)
title('ROI\gamma')
set(gca,'fontsize',fonte)
xlabel('time (s)')

eeg = Matrix_eeg_C(:,:,10);
teeg = (0:0.01:9.99);

passoin = round(passoin/100);
passofin = round(passofin/100);
figure(2)
subplot(2,2,1)
plot(teeg(passoin:passofin),eeg(3,passoin:passofin),'k','linewidth',linea)
title('ROI\theta')
set(gca,'fontsize',fonte)
%xlabel('time (s)')
ylabel('mean field potential')
subplot(222)
plot(teeg(passoin:passofin),eeg(4,passoin:passofin),'k','linewidth',linea)
title('ROI\alpha')
set(gca,'fontsize',fonte)
%xlabel('time (s)')
subplot(223)
plot(teeg(passoin:passofin),eeg(1,passoin:passofin),'k','linewidth',linea)
title('ROI\beta')
set(gca,'fontsize',fonte)
xlabel('time (s)')
ylabel('mean field potential')
subplot(224)
plot(teeg(passoin:passofin),eeg(2,passoin:passofin),'k','linewidth',linea)
title('ROI\gamma')
set(gca,'fontsize',fonte)
xlabel('time (s)')

zeropadding = 1000; 
fs = 100;
window = 50;  
figure(3)
subplot(2,2,1)
[Peeg,f] =  pwelch(eeg(3,:),window,[],zeropadding,fs);
plot(f(30:end),Peeg(30:end),'k','linewidth',2)
title('ROI\theta')
set(gca,'fontsize',fonte)
ylabel('Power spectral density')
xlim([0 50])
%xlabel('frequency (Hz)')
subplot(222)
[Peeg,f] =  pwelch(eeg(4,:),window,[],zeropadding,fs);
plot(f(30:end),Peeg(30:end),'k','linewidth',2)
title('ROI\alpha')
set(gca,'fontsize',fonte)
%xlabel('frequency (Hz)')
xlim([0 50])
subplot(223)
[Peeg,f] =  pwelch(eeg(1,:),window,[],zeropadding,fs);
plot(f(30:end),Peeg(30:end),'k','linewidth',2)
title('ROI\beta')
ylabel('Power spectral density')
xlabel('frequency (Hz)')
set(gca,'fontsize',fonte)
xlim([0 50])
subplot(224)
[Peeg,f] =  pwelch(eeg(2,:),window,[],zeropadding,fs);
plot(f(30:end),Peeg(30:end),'k','linewidth',2)
title('ROI\gamma')
set(gca,'fontsize',fonte)
xlabel('frequency (Hz)')
xlim([0 50])










