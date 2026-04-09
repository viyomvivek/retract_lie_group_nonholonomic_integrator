%% Nonholonomic Rigid Body SO(3) Left Exponential 

clc; clear; close all;

%% -------------------------
% Parameters
%% -------------------------
t = 0.01;
I = diag([1 10 100]);
I_inv = diag([1 0.1 0.01]);

N = 1800;
tol = 1e-10;
maxfun = 100;

%% -------------------------
% Initial conditions
%% -------------------------
R0 = eye(3);
pi0 = [1; 1; 0];     % constrained initial condition

%% -------------------------
% Preallocation
%% -------------------------
R = zeros(3,3,N+1);
pi = zeros(3,N+1);
omega_sol = zeros(2,N);     % only 2 DOF

energy = zeros(1,N+1);
constraint_error = zeros(1,N+1);
orthoerror = zeros(1,N+1);

R(:,:,1) = R0;
pi(:,1) = pi0;

%% -------------------------
% fsolve options
%% -------------------------
opts = optimoptions('fsolve', ...
    'Display','off', ...
    'FunctionTolerance', tol, ...
    'MaxFunctionEvaluations', 2000, ...
    'MaxIterations', maxfun);

%% -------------------------
% Initial metrics
%% -------------------------
energy(1) = E_fun(pi(:,1), I_inv);
constraint_error(1) = pi(3,1);
orthoerror(1) = O_fun(R(:,:,1));

%% -------------------------
% Main loop
%% -------------------------
for k = 1:N
    Rk = R(:,:,k);
    pik = pi(:,k);

    % initial guess
    if k == 1
        omega0 = ones(2,1);
    else
        omega0 = omega_sol(:,k-1);
    end

    % implicit solve
    g_step = @(omega2) g_fun(omega2, pik, I, t);

    try
        [omega2k, ~, exitflag] = fsolve(g_step, omega0, opts);
    catch ME
        warning('fsolve error at step %d: %s', k, ME.message);
        exitflag = -1;
    end

    if exitflag <= 0
        warning('fsolve failed at step %d. Using fallback.', k);
        omega2k = omega0;
    end

    omega_sol(:,k) = omega2k;

    % reconstruct full omega
    omegak = [omega2k; 0];

    % update
    R_next = f_fun(Rk, omegak, t);
    pi_next = h_fun(omegak, pik, t);

    % enforce constraint (projection)
    pi_next(3) = 0;

    R(:,:,k+1) = R_next;
    pi(:,k+1) = pi_next;

    % diagnostics
    energy(k+1) = E_fun(pi_next, I_inv);
    constraint_error(k+1) = pi_next(3);
    orthoerror(k+1) = O_fun(R_next);
end

kvec = 0:N;

figure;

subplot(3,1,1);
plot(kvec, energy - energy(1), 'LineWidth',1.5);
xlabel('k');
ylabel('Energy Error');
title('Energy Error (Exponential)');
grid on;

subplot(3,1,2);
plot(kvec, constraint_error, 'LineWidth',1.5);
xlabel('k');
ylabel('\pi_3');
title('Constraint Error');
grid on;

subplot(3,1,3);
plot(kvec, orthoerror, 'LineWidth',1.5);
xlabel('k');
ylabel('Orthogonality Error');
title('Orthogonality Error');
grid on;

%% ============================================================
%% FUNCTIONS
%% ============================================================

% --- Group update (Exponential map)
function Rnext = f_fun(Rcur, omega, t)
    Rnext = Rcur * expm(t * hat(omega));
end

% --- Momentum update (coadjoint action)
function pinext = h_fun(omega, picur, t)
    pinext = expm(t * hat(omega)) * picur;
end

% --- Implicit equation
function res = g_fun(omega2, picur, I, t)

    omega = [omega2; 0];    % constraint built-in
    omega_hat = hat(omega);

    theta = norm(t * omega);

    if theta < 1e-8
        a = 1/2;
        b = 1/6;
        c = 1;
    else
        a = (1 - cos(theta)) / theta^2;
        b = (theta - sin(theta)) / theta^3;
        c = sin(theta) / theta;
    end

    full_res = (eye(3) - a*t*omega_hat + b*t^2*omega_hat^2) * ...
               (eye(3) + c*t*omega_hat + a*t^2*omega_hat^2) * picur ...
               - t * I * omega;

    res = full_res(1:2);   % only unconstrained directions
end

% --- hat map
function S = hat(omega)
    S = [0 -omega(3) omega(2);
         omega(3) 0 -omega(1);
         -omega(2) omega(1) 0];
end

% --- Energy
function val = E_fun(pi, I_inv)
    val = 0.5 * pi' * I_inv * pi;
end

% --- Orthogonality error
function val = O_fun(R)
    val = norm(eye(3) - R*R');
end