clc; clear all; close all;
I=imread('lena.bmp');
delta=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Insertion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                          dwt
[cA,cH,cV,cD] = dwt2(I,'db2');
%%                          Generate messages and positions
d=2;f=255;
N=(f-d+1)^2;
Nt=N/2;

[ messages_mtx,m,M,pos ] = gen_msg( Nt,N );

%%                          Insertion
cH1=cH;cV1=cV;cD1=cD;
cH1(d:f,d:f)=ins_Q(cH(d:f,d:f),messages_mtx, delta);
cV1(d:f,d:f)=ins_Q(cV(d:f,d:f),messages_mtx, delta);
cD1(d:f,d:f)=ins_Q(cD(d:f,d:f),messages_mtx, delta);
%%                          Inverse dwt
I1=uint8(idwt2(cA,cH1,cV1,cD1,'db2'));
disp(['PSNR: ' num2str(psnr(I,I1))] )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I1=attack(I1,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                             IDWT
[cA2,cH2,cV2,cD2] = dwt2(I1,'db2');
%%                              prepare samples
[ m_host_cD,M_host_cD ] = dec_msg( cD2(d:f,d:f),pos,Nt,N );
[ m_host_cH,M_host_cH ] = dec_msg( cH2(d:f,d:f),pos,Nt,N );
[ m_host_cV,M_host_cV ] = dec_msg( cV2(d:f,d:f),pos,Nt,N );

xm(:,1)=[m_host_cH];%message connu
xm(:,2)=[m_host_cV];%message connu
xm(:,3)=[m_host_cD];%message connu

ym(:,1)=[m];% known message


xM(:,1)=[M_host_cH];%message unconnu
xM(:,2)=[M_host_cV];%message unconnu
xM(:,3)=[M_host_cD];%message unconnu


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               SVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              SVM training
SVMModel = fitcsvm(xm,ym,'KernelFunction','polynomial');% rbf linear polynomial gaussian
sv = SVMModel.SupportVectors;


%%                             


[label,score] = predict(SVMModel,xM); %SVM tuning
[number_svm,ratio_svm] = biterr(label,M');

mean_M=round((det_Q(M_host_cH,delta)+det_Q(M_host_cV,delta)+det_Q(M_host_cD,delta))/3);%valeur moyenne des 3 sous bandes
[number,ratio] = biterr(mean_M,M);

disp(['Extraction SVM: ' num2str(ratio_svm) ' with ' num2str(number_svm) ' erroneous bits'] )
disp(['Extraction QIM: ' num2str(ratio) ' with ' num2str(number) ' erroneous bits'] )




