%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 1
Number of samples = 400; class means [0,0]' and [3,3]' ; 
class covariance matrices both set to I; equal class priors.
%}
clear all
close all
P1=0.5;
P2=0.5;
cov1=[1 0;0 1];
cov2=[1 0;0 1];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(1),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 1
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 2
All parameters same as (1), 
but both covariance matrices changed to [3 1;1 0.8].
%}
P1=0.5;
P2=0.5;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(2),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 2
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 3
Number of samples = 400;
 class means [0,0]' and [2,2]' ;
 class covariance matrices [2 0.5;0.5 1]
and [2 -1.9;-1.9 5]; equal class priors.

%}
P1=0.5;
P2=0.5;
cov1=[2 0.5;0.5 1];
cov2=[1 -1.9;-1.9 5];
mu1=[0;0];
mu2=[2;2];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(3),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 3
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 4
Same (1), but prior for class priors are 0:05 and 0:95.

%}
P1=0.05;
P2=0.95;
cov1=[1 0;0 1];
cov2=[1 0;0 1];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(4),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 4
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 5
Same (2), but prior for class priors are 0:05 and 0:95.
%}
P1=0.05;
P2=0.95;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(5),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 5
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 3
Part 6
Same (3), but prior for class priors are 0:05 and 0:95.

%}
P1=0.05;
P2=0.95;
cov1=[2 0.5;0.5 1];
cov2=[1 -1.9;-1.9 5];
mu1=[0;0];
mu2=[2;2];
j=1;
k=1;
N1=0;
N2=0;
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(6),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
subplot(2,1,2), 
plot(y1(1,:),zeros(1,N1),'r*');
hold on;
plot(y2(1,:),zeros(1,N2),'bo');
axis equal;
db=(mu1hat+mu2hat)/2;
y = w'*db;
xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1','Class 2','Decision boundary');
grid on
% end of part 6

