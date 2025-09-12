
n_weights = 2; % Define the number of weights
n_samples = 2;
a=-5;
b=5;
A = 1;
eta = 1e-3; 
syms t [1 n_samples] real;
syms beta_x [1 n_weights] real;
syms beta_y [1 n_weights] real;

W_random = a+(b-a)*randn(1,n_weights);
b_random = a+(b-a)*randn(1,n_weights);
H = vpa(tanh(t'*W_random+b_random),5);
dH_dt = vpa(diff_over_time_vector(H,t'),5);
d2H_d2t = vpa(diff_over_time_vector(dH_dt,t'),5);

%% Define instantiated states and derivatives
x0 = vpa(H*beta_x',5);
y0 = vpa(H*beta_y',5);
%% Define new time derivative of states
dx0_dt = vpa(diff_over_time_vector(x0,t'),5);
dy0_dt = vpa(diff_over_time_vector(y0,t'),5);
%% Define constraints lhs
c1_1_lhs = vpa(dx0_dt - dy0_dt,5);
c2_1_lhs = vpa(dx0_dt + dy0_dt,5);
c3_1_lhs = vpa(-dx0_dt + dy0_dt,5);
c4_1_lhs = vpa(-dx0_dt - dy0_dt,5);
%% Find smooth penalties and updates
phi_1_1 = vpa(smooth_penalty(less_than_inequality(c1_1_lhs,a)),5);
phi_1_2 = vpa(smooth_penalty(less_than_inequality(c2_1_lhs,a)),5);
phi_1_3 = vpa(smooth_penalty(less_than_inequality(c3_1_lhs,a)),5);
phi_1_4 = vpa(smooth_penalty(less_than_inequality(c4_1_lhs,a)),5);

dphi_1_1_d_beta_x = vpa(diff_over_beta_vector(phi_1_1,beta_x'),5);
dphi_1_2_d_beta_x = vpa(diff_over_beta_vector(phi_1_2,beta_x'),5);
dphi_1_3_d_beta_x = vpa(diff_over_beta_vector(phi_1_3,beta_x'),5);
dphi_1_4_d_beta_x = vpa(diff_over_beta_vector(phi_1_4,beta_x'),5);

dphi_1_1_d_beta_y = vpa(diff_over_beta_vector(phi_1_1,beta_y'),5);
dphi_1_2_d_beta_y = vpa(diff_over_beta_vector(phi_1_2,beta_y'),5);
dphi_1_3_d_beta_y = vpa(diff_over_beta_vector(phi_1_3,beta_y'),5);
dphi_1_4_d_beta_y = vpa(diff_over_beta_vector(phi_1_4,beta_y'),5);
%% Define updates 
update_phi_1_1_x = vpa(bounded_tanh(dphi_1_1_d_beta_x, A, eta), 5);
update_phi_1_2_x = vpa(bounded_tanh(dphi_1_2_d_beta_x, A, eta), 5);
update_phi_1_3_x = vpa(bounded_tanh(dphi_1_3_d_beta_x, A, eta), 5);
update_phi_1_4_x = vpa(bounded_tanh(dphi_1_4_d_beta_x, A, eta), 5);

update_phi_1_1_y = vpa(bounded_tanh(dphi_1_1_d_beta_y, A, eta), 5);
update_phi_1_2_y = vpa(bounded_tanh(dphi_1_2_d_beta_y, A, eta), 5);
update_phi_1_3_y = vpa(bounded_tanh(dphi_1_3_d_beta_y, A, eta), 5);
update_phi_1_4_y = vpa(bounded_tanh(dphi_1_4_d_beta_y, A, eta), 5);

%% Sum all updates
update_phi_1_x= vpa(update_phi_1_1_x+update_phi_1_2_x+update_phi_1_3_x+update_phi_1_4_x,5);
update_phi_1_y= vpa(update_phi_1_1_y+update_phi_1_2_y+update_phi_1_3_y+update_phi_1_4_y,5);
%% Update states and find new derivatives
x1 = vpa(H*(beta_x-update_phi_1_x)',5);
y1 = vpa(H*(beta_y-update_phi_1_y)',5);
%% Define new time derivative of states
dx1_dt = vpa(diff_over_time_vector(x1,t'),5);
dy1_dt = vpa(diff_over_time_vector(y1,t'),5);
%% Define constraints lhs (with VPA)
c1_2_lhs = vpa(dx1_dt - dy1_dt, 5);
c2_2_lhs = vpa(dx1_dt + dy1_dt, 5);
c3_2_lhs = vpa(-dx1_dt + dy1_dt, 5);
c4_2_lhs = vpa(-dx1_dt - dy1_dt, 5);

%% Find smooth penalties and updates (with VPA)
phi_2_1 = vpa(smooth_penalty(less_than_inequality(c1_2_lhs, a)), 5);
phi_2_2 = vpa(smooth_penalty(less_than_inequality(c2_2_lhs, a)), 5);
phi_2_3 = vpa(smooth_penalty(less_than_inequality(c3_2_lhs, a)), 5);
phi_2_4 = vpa(smooth_penalty(less_than_inequality(c4_2_lhs, a)), 5);

dphi_2_1_d_beta_x = vpa(diff_over_beta_vector(phi_2_1, beta_x'), 5);
dphi_2_2_d_beta_x = vpa(diff_over_beta_vector(phi_2_2, beta_x'), 5);
dphi_2_3_d_beta_x = vpa(diff_over_beta_vector(phi_2_3, beta_x'), 5);
dphi_2_4_d_beta_x = vpa(diff_over_beta_vector(phi_2_4, beta_x'), 5);

dphi_2_1_d_beta_y = vpa(diff_over_beta_vector(phi_2_1, beta_y'), 5);
dphi_2_2_d_beta_y = vpa(diff_over_beta_vector(phi_2_2, beta_y'), 5);
dphi_2_3_d_beta_y = vpa(diff_over_beta_vector(phi_2_3, beta_y'), 5);
dphi_2_4_d_beta_y = vpa(diff_over_beta_vector(phi_2_4, beta_y'), 5);

%% Define updates 
update_phi_2_1_x = vpa(bounded_tanh(dphi_2_1_d_beta_x, A, eta), 5);
update_phi_2_2_x = vpa(bounded_tanh(dphi_2_2_d_beta_x, A, eta), 5);
update_phi_2_3_x = vpa(bounded_tanh(dphi_2_3_d_beta_x, A, eta), 5);
update_phi_2_4_x = vpa(bounded_tanh(dphi_2_4_d_beta_x, A, eta), 5);

update_phi_2_1_y = vpa(bounded_tanh(dphi_2_1_d_beta_y, A, eta), 5);
update_phi_2_2_y = vpa(bounded_tanh(dphi_2_2_d_beta_y, A, eta), 5);
update_phi_2_3_y = vpa(bounded_tanh(dphi_2_3_d_beta_y, A, eta), 5);
update_phi_2_4_y = vpa(bounded_tanh(dphi_2_4_d_beta_y, A, eta), 5);

%% Sum all updates with VPA
update_phi_2_x = vpa(update_phi_2_1_x + update_phi_2_2_x + update_phi_2_3_x + update_phi_2_4_x, 5);
update_phi_2_y = vpa(update_phi_2_1_y + update_phi_2_2_y + update_phi_2_3_y + update_phi_2_4_y, 5);

%% Update states and find new derivatives
x2 = vpa(H*(beta_x-update_phi_1_x-update_phi_2_x)',5);
y2 = vpa(H*(beta_y-update_phi_1_y-update_phi_2_y)',5);
%% Define new time derivative of states
dx2_dt = vpa(diff_over_time_vector(x2,t'),5);
dy2_dt = vpa(diff_over_time_vector(y2,t'),5);

%% Functions here
function y = bounded_tanh(f,A,eta)
    y = eta*A*tanh(f/A*eta);
end 
function y = less_than_inequality(f,a)
    y = f-a;
end 
function y = more_than_penalty(f,a)
    y = -f-a;
end
function y = smooth_penalty(f)
    y = exp(f);
end

function y=diff_over_time_vector(f,t)
    y = zeros(size(t));
    for i = 1:size(t,1)
        y =vpa(diff(f,t(i)),5)+y;
        display(i)
        display('Did one diff step over time vector')
    end 
end

function y=diff_over_beta_vector(f,beta)
    y = vpa(diff(f,beta(1)),5);
    for i = 2:size(beta,1)
        y(:,i)= vpa(diff(f,beta(i)),5);

        display('Did one diff step over beta vector')
    end
    y = sum(y,1);
end