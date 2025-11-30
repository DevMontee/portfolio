clc;
clear
close all;
%inputs
mf = 80;
AR = 7.75;
S = 17.6;
me = (0.4535*2.5*(10.7639*S))/2;
m = me+mf;
Span = sqrt(S*AR);
l = Span/2;
c = Span/AR;
b = c/2;
e= -0.3;
a= -0.2;
xtheta = e-a;
rho = 1.225;
mu = m/(pi*rho*b^2*l);
EI = 200000;
GJ = 100000;
Ip = 7;
rs = Ip/(m*b^2);
wh = 1.8751^2*sqrt(EI/(m*l^3));
wt = (pi/2)*sqrt(GJ/(Ip*l));
sigma = wh/wt;
N = 300; %number of divisions
%equations and pre-defining vectors/matrices

V_vec = linspace(0,3,N); %the reduced velocity vector
r1 = zeros(1,N); %first root
r2 = zeros(1,N); %second root
r3 = zeros(1,N); %third root
r4 = zeros(1,N); %fourth root
k_mat = zeros(4,N);
damp_mat = zeros(4,N);
omega_ratio_mat = zeros(4,N);
k_guess = 1;
tol = 1e-6;
for j = 2 : N
V = V_vec (j);
k = k_guess;
for jj = 1 : 4
diff_k = 1;
while abs(diff_k) > tol
Cfun = (0.01365+0.2808*1i*k-k^2/2)/(0.01365+0.3455*1i*k-k^2); 
f11 = sigma^2/V^2 - k^2/mu + 2*1i*k*Cfun/mu;
f12 = (k*(1i+a*k)+(2+1i*k*(1-2*a))*Cfun)/mu;
f21 = (a*k^2-1i*k*(1+2*a)*Cfun)/mu;
f22 = ( 8*mu*rs/V^2 + 4*1i*(1+2*a)*(2*1i-k*(1-2*a))*Cfun-k*(k-4*1i+8*a*(1i+a*k)) )/(8*mu);
poly = [rs-xtheta^2 0 f22+f11*rs-f21*xtheta-f12*xtheta 0 f11*f22-f12*f21]; 
rts = roots( poly );
[~,indx] = sort(imag(rts));
rts_sorted = rts(indx);
rootjj = rts_sorted(jj);
k_new = imag(rootjj);
damp_new = real(rts_sorted(jj));
diff_k = k_new - k;
k = k_new;
disp(diff_k);
end;
k_mat(jj,j) = k_new;
damp_mat(jj,j) = damp_new;
omega_ratio_mat(jj,j) = k_new*V;
end;
end;
figure;
plot(V_vec,omega_ratio_mat(1,:),'bo');
hold on;
plot(V_vec,omega_ratio_mat(2,:),'ko');
plot(V_vec,omega_ratio_mat(3,:),'ro');
plot(V_vec,omega_ratio_mat(4,:),'go');
grid on;
xlabel('V= U/(b \omega_\theta)');
ylabel('(\omega/\omega_\theta)');
figure;
plot(V_vec,damp_mat(1,:),'bs');
hold on;
plot(V_vec,damp_mat(2,:),'ks');
plot(V_vec,damp_mat(3,:),'rs');
plot(V_vec,damp_mat(4,:),'gs');
grid on;
xlabel('V= U/(b \omega_\theta)');
ylabel('\gamma');