%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
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
I=[1,0;0,1];
mu=[0;0];
[A1,num1] = cholcov(cov1);
[A2,num2] = cholcov(cov2);
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
        G=mvnrnd(mu,I);
        G=G';
        x(:,i)= A1*G + mu1;
        label(i)=1;
    else 
        G=mvnrnd(mu,I);
        G=G';
        x(:,i)= A2*G + mu2;
        label(i)=2;
    end
end
label=label';
x=x';
figure(1);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part1=r
probability_of_error_part1=(r/400)
%End of part 1
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
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
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=1;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=2;
    end
end
label=label';
x=x';
figure(2);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part2=r
probability_of_error_part2=(r/400)
%End of part 2
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
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
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=1;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=2;
    end
end
label=label';
x=x';
figure(3);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part3=r
probability_of_error_part3=(r/400)
%End of part 3
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
Part 4
Same (1), but prior for class priors are 0:05 and 0:95.

%}
P1=0.05;
P2=0.95;
cov1=[1 0;0 1];
cov2=[1 0;0 1];
mu1=[0;0];
mu2=[3;3];
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=1;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=2;
    end
end
label=label';
x=x';
figure(4);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part4=r
probability_of_error_part4=(r/400)
%End of part 4
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
Part 5
Same (2), but prior for class priors are 0:05 and 0:95.
%}
P1=0.05;
P2=0.95;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=1;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=2;
    end
end
label=label';
x=x';
figure(5);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part5=r
probability_of_error_part5=(r/400)
%End of part 5
clear all
%{
Homework 2 Intro to ML and Pattern Recognition
EECE 5644 Fall 2019
Question 2
Part 6
Same (3), but prior for class priors are 0:05 and 0:95.

%}
P1=0.05;
P2=0.95;
cov1=[2 0.5;0.5 1];
cov2=[1 -1.9;-1.9 5];
mu1=[0;0];
mu2=[2;2];
for i=1:400
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=1;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=2;
    end
end
label=label';
x=x';
figure(6);
subplot(1,2,1);
for i=1:400
    if label(i)==1
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:400
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=1;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=2;
    end
end
subplot(1,2,2);
plot(q1(:,1),q1(:,2),'r+');
hold on
plot(q2(:,1),q2(:,2),'b*');
title('Infered Scatter plot of data');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];
z=eq(x,q);
r=0;
h=h';
z=eq(label,h);
for i=1:400
    if z(i)==0
        r=r+1;
    end
end
number_of_errors_part6=r
probability_of_error_part6=(r/400);
%End of part 6
