%M = me + mf 

clc;
clear
close all;

%inputs
mf = 0;
AR = 7.75;
S = 17.6;
me = (0.4535*2.5*(10.7639*S))/2;
m = me+mf;
Span = sqrt(S*AR);
l = Span/2;
c = Span/AR;
b = c/2;
e= -0.1;
a= -0.2;
xtheta = e-a;
rho = 1.225; % change the value according to the altitude 
mu = m/(pi*rho*b^2*l);
EI = 200000;
GJ = 100000;
Ip = 4;
rs = Ip/(m*b^2);
wh = 1.8751^2*sqrt(EI/(m*l^3));
wt = (pi/2)*sqrt(GJ/(Ip*l));
sigma = wh/wt;


%equations and pre-setting

V_vec = 0:0.05:4; %the reduced velocity vector
N = length(V_vec); %number of divisions
r1 = zeros(1,N); %first root
r2 = zeros(1,N); %second root
r3 = zeros(1,N); %third root
r4 = zeros(1,N); %fourth root

%for-loop
for i = 2 : N
    V = V_vec( i );
    p = [rs-xtheta^2 0 (rs/V^2)-(2/mu)*(a+0.5)+(sigma^2*rs/V^2)-2*xtheta/mu 0 ...
        (sigma/V)^2*(rs/V^2-(2/mu)*(a+0.5))];
    rts = roots(p); %finding roots of the polynomial. These roots can have real and imaginary parts
    r1(i) = rts(1); %1st root
    r2(i) = rts(2); %2nd root
    r3(i) = rts(3); %3rd root
    r4(i) = rts(4); %4th root
end

%plot dimensionless frequency versus dimensionless velocity
figure;
plot(V_vec,V_vec.*imag(r1),'ro','MarkerSize',10); % Note that .* is used for the element-wise
%multiplication."V_vec" is multiplied element-wise by "imag(r1)", for example,
%to normalize the values of "imag(r1)" with respect to "omega_theta"; refer
%to equations (5.33) and (5.34) in the textbook by Hodges and Pierce (2011).
hold on;
plot(V_vec,V_vec.*imag(r2),'bd','MarkerSize',8);
plot(V_vec,V_vec.*imag(r3),'k*','MarkerSize',6);
plot(V_vec,V_vec.*imag(r4),'gs','MarkerSize',4);
xlabel('V');
ylabel('\Omega/\omega_\theta');
grid on;
hold off;

%plot dimensionless damping versus dimensionless velocity
figure;
plot(V_vec,V_vec.*real(r1),'ro','MarkerSize',10);
hold on;
plot(V_vec,V_vec.*real(r2),'bd','MarkerSize',8);
plot(V_vec,V_vec.*real(r3),'k*','MarkerSize',6);
plot(V_vec,V_vec.*real(r4),'gs','MarkerSize',4);
xlabel('V');
ylabel('\Gamma/\omega_\theta');
grid on;
hold off;

