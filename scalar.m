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

[ messages_mtx_cA,m_cA,M_cA,pos_cA ] = gen_msg( Nt,N );
[ messages_mtx_cH,m_cH,M_cH,pos_cH ] = gen_msg( Nt,N );
[ messages_mtx_cV,m_cV,M_cV,pos_cV ] = gen_msg( Nt,N );
[ messages_mtx_cD,m_cD,M_cD,pos_cD ] = gen_msg( Nt,N );
%%                          Insertion
cH1=cH;cV1=cV;cD1=cD;
cH1(d:f,d:f)=ins_Q(cH(d:f,d:f),messages_mtx_cH, delta);
cV1(d:f,d:f)=ins_Q(cV(d:f,d:f),messages_mtx_cV, delta);
cD1(d:f,d:f)=ins_Q(cD(d:f,d:f),messages_mtx_cD, delta);
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
[ m_host_cD,M_host_cD ] = dec_msg( cD2(d:f,d:f),pos_cD,Nt,N );
[ m_host_cH,M_host_cH ] = dec_msg( cH2(d:f,d:f),pos_cH,Nt,N );
[ m_host_cV,M_host_cV ] = dec_msg( cV2(d:f,d:f),pos_cV,Nt,N );

xm(:,1)=[m_host_cH,m_host_cV,m_host_cD];%message connu
xm(:,2)=1;
ym(:,1)=[m_cH,m_cV,m_cD];% value

xM(:,1)=[M_host_cH,M_host_cV,M_host_cD];% message to predict
xM(:,2)=1;
msg(:,1)=[M_cH,M_cV,M_cD];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               SVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              SVM training
SVMModel = fitcsvm(xm,ym,'KernelFunction','gaussian');% rbf linear polynomial gaussian
sv = SVMModel.SupportVectors;
figure
gscatter(xm(:,1),xm(:,2),ym)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('wahed','sfer','Support Vector')
hold off

%%                             


[label,score] = predict(SVMModel,xM); %SVM tuning
[number_svm,ratio_svm] = biterr(label,msg);

[number,ratio] = biterr(det_Q(xM(:,1),delta),msg);% classical extraction

disp(['Extraction SVM: ' num2str(ratio_svm) ' with ' num2str(number_svm) ' erroneous bits'] )
disp(['Extraction QIM: ' num2str(ratio) ' with ' num2str(number) ' erroneous bits'] )




