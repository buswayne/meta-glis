%% Bayesian optimization

addpath('export_fig');
addpath('robots');

%% Cleanup %%

clear;
close all;
clc;

%% Startup %%

rng(42); % set seed for reproducibility
addpath('util');

%% Constants  definition %% 
Ts = 1e-3;         % sampling time (s). We assume measurements are available at this rate
Tsim = 4;         % simulation length (s)

%% robot

n_DoFs = 7;

friction = [2 2 2 2 2 2 2];

Robot = panda_robot();

%% Motion Reference

t0 = 0; % [s]
tf = 2.; % [s]
time = t0:Ts:Tsim; % [s]

% R_0 = diag([1,1,1]);
% 
% x_0=[0.3 0. 0.5];  
% x_f=[0.4 0. 0.6];
% 
% dx_0=[0 0 0];
% dx_f=[0 0 0];
% 
% ddx_0=[0 0 0];
% ddx_f=[0 0 0];
% 
% T_0 = diag([1,1,1,1]);
% T_0(1:3,1:3) = R_0;
% T_0(1:3,4) = x_0(1:3)';
% 
% T_robot(:,:,1) = T_0;
% 
% q_0 = Robot.ikine(T_0,[-0.7160   -0.5850    0.3504   -1.5666    0.2241   -2.1201   -2.8398]);
% 
% xM = [x_0;x_f;dx_0;dx_f;ddx_0;ddx_f];
% 
% C = [1   t0  t0^2    t0^3     t0^4    t0^5
%      1   tf  tf^2    tf^3     tf^4    tf^5
%      0   1   2*t0    3*t0^2   4*t0^3  5*t0^4
%      0   1   2*tf    3*tf^2   4*tf^3  5*tf^4
%      0   0   2       6*t0     12*t0^2 20*t0^3
%      0   0   2       6*tf     12*tf^2 20*tf^3];
% 
% a=C^-1*xM;
% 
% q_r(1,:) = q_0;
% dq_r(1,:) = [0, 0, 0, 0, 0, 0, 0];
% ddq_r(1,:) = [0, 0, 0, 0, 0, 0, 0];
% 
% x_r(1,:) = x_0;
% dx_r(1,:) = [0, 0, 0];
% ddx_r(1,:) = [0, 0, 0];
% 
% disp('calculating trj...');
% 
% wb = waitbar(0,'Please wait...');
% 
% for jj=2:length(time)
% 
%     if (time(jj)<=tf)
% 
%         t_i = time(jj);
% 
%         x_r(jj,1)=a(1,1)+a(2,1)*t_i+a(3,1)*t_i^2+a(4,1)*t_i^3+a(5,1)*t_i^4+a(6,1)*t_i^5;
%         x_r(jj,2)=a(1,2)+a(2,2)*t_i+a(3,2)*t_i^2+a(4,2)*t_i^3+a(5,2)*t_i^4+a(6,2)*t_i^5;
%         x_r(jj,3)=a(1,3)+a(2,3)*t_i+a(3,3)*t_i^2+a(4,3)*t_i^3+a(5,3)*t_i^4+a(6,3)*t_i^5;
% 
%         dx_r(jj,1)=a(2,1)+2*a(3,1)*t_i+3*a(4,1)*t_i^2+4*a(5,1)*t_i^3+5*a(6,1)*t_i^4;
%         dx_r(jj,2)=a(2,2)+2*a(3,2)*t_i+3*a(4,2)*t_i^2+4*a(5,2)*t_i^3+5*a(6,2)*t_i^4;
%         dx_r(jj,3)=a(2,3)+2*a(3,3)*t_i+3*a(4,3)*t_i^2+4*a(5,3)*t_i^3+5*a(6,3)*t_i^4;
% 
%         ddx_r(jj,1)=2*a(3,1)+6*a(4,1)*t_i+12*a(5,1)*t_i^2+20*a(6,1)*t_i^3;
%         ddx_r(jj,2)=2*a(3,2)+6*a(4,2)*t_i+12*a(5,2)*t_i^2+20*a(6,2)*t_i^3;
%         ddx_r(jj,3)=2*a(3,3)+6*a(4,3)*t_i+12*a(5,3)*t_i^2+20*a(6,3)*t_i^3;
% 
%         T_robot(:,:,jj)=[R_0 [x_r(jj,1:3)'];zeros(1,3) 1];
% 
%         q_r(jj,:) = Robot.ikine(T_robot(:,:,jj),q_r(jj-1,:));
%         dq_r(jj,:) = (q_r(jj,:) - q_r(jj-1,:))./Ts;
%         ddq_r(jj,:) = (dq_r(jj,:) - dq_r(jj-1,:))./Ts;
% 
%     else
% 
%         x_r(jj,:) = x_r(jj-1,:);
%         dx_r(jj,:) = [0, 0, 0];
%         ddx_r(jj,:) = [0, 0, 0];
% 
%         q_r(jj,:) = q_r(jj-1,:);
%         dq_r(jj,:) = [0, 0, 0, 0, 0, 0, 0];
%         ddq_r(jj,:) = [0, 0, 0, 0, 0, 0, 0];
% 
%     end
% 
% waitbar(jj/length(time),wb);
% 
% end
% 
% close(wb)
% 
% disp('going to optimization...');
% 
% r.x_r = x_r;
% r.dx_r = dx_r;
% r.ddx_r = ddx_r;
% r.q_r = q_r;
% r.dq_r = dq_r;
% r.ddq_r = ddq_r;
% 

toll_qerr = 20*pi/180;
A = 15*pi/180;
f = 1;
q_0 = [-0.7160   -0.5850    0.3504   -1.5666    0.2241   -2.1201   -2.8398];
q_r(1,:) = q_0-A;
dq_r(1,:) = [0, 0, 0, 0, 0, 0, 0];
ddq_r(1,:) = [0, 0, 0, 0, 0, 0, 0];
for jj=2:length(time)
    q_r(jj,:) = q_0 + A * sin(2 * pi * f * time(jj) - pi / 2);
    dq_r(jj,:) = (q_r(jj,:) - q_r(jj-1,:))./Ts;
    ddq_r(jj,:) = (dq_r(jj,:) - dq_r(jj-1,:))./Ts;
end

r.q_r = q_r;
r.dq_r = dq_r;
r.ddq_r = ddq_r;

%%
% all constants in a structure for convenience 

const.Ts = Ts; % inner loop sampling time
const.Tsim = Tsim;
const.time = time;

const.n_DoFs = n_DoFs;

const.r = r;

const.Robot = Robot;
const.Robot_friction = friction;
const.q_0 = q_0;
const.toll_qerr = toll_qerr;

%% Bayesian optimization of controller parameters 
% Define bounds on the optimization variables  %

opt_vars = [];

bound_min_Kp = 0.;
bound_max_Kp = 300;

bound_min_Kd = 0;
bound_max_Kd = 50;

bound_min_Ki = 0.;
bound_max_Ki = 500.;

%%
% PID parameters
opt_vars = [opt_vars optimizableVariable('Kp1', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki1', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd1', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp2', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki2', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd2', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp3', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki3', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd3', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp4', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki4', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd4', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp5', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki5', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd5', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp6', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki6', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd6', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative
opt_vars = [opt_vars optimizableVariable('Kp7', [bound_min_Kp, bound_max_Kp],'Type','real')]; % PID proportional
opt_vars = [opt_vars optimizableVariable('Ki7', [bound_min_Ki, bound_max_Ki],'Type','real')]; % PID integral
opt_vars = [opt_vars optimizableVariable('Kd7', [bound_min_Kd, bound_max_Kd],'Type','real')]; % PID derivative

%%
% Define objective function as a function of the optimization variables
% opt_vars only, to be passed to the optimizer

func =  @(x_vars)(obj_PID_panda(x_vars,const));
initial_X = []; % array2table(rho_init, 'VariableNames', opt_var_names);

clear obj_PID_panda; % just to reset the function inner counter

%%
% Perform bayesian optimization 

optimize = true;

if optimize
    results = bayesopt(func,opt_vars,...
        'Verbose',0,...
        'AcquisitionFunctionName','expected-improvement',... %-plus',...
        'IsObjectiveDeterministic', true,... % simulations with noise --> objective function is not deterministic
        'MaxObjectiveEvaluations', 100,...
        'MaxTime', inf,...
        'NumCoupledConstraints',0, ...
        'NumSeedPoint',10,...
        'GPActiveSetSize', 300,...
        'PlotFcn',{@plotMinObjective,@plotObjectiveEvaluationTime}); %);
        %,...   'NumSeedPoints',10,);
        %'XConstraintFcn', @norm_constraint,...
else
    load results
end

best_vars = results.bestPoint;

%%
% Evaluate performance of the optimal design
close all

figure
obj_PID_panda(best_vars,const);

%%

const.plot_fig = true;
const.Tsim = 10;
N_mc = 5;
OBJ_VEC = zeros(N_mc,1);
for i=1:N_mc
    OBJ_VEC(i) = obj_PID_panda(best_vars,const);
    disp(OBJ_VEC(i));
end

%%

const.Tsim = 10;
idx_min = results.IndexOfMinimumTrace(end);
results.XTrace(idx_min,:)
clear obj_PID_panda;
% obj_val=obj_PID_panda(results.XTrace(idx_min,:), const, idx_min);

%%

idx_min = results.IndexOfMinimumTrace(end);
N = length(results.ObjectiveTrace);
iteration = 1:N;
f = figure(2);
plot(iteration, results.ObjectiveTrace, 'k*')
hold on;
plot(results.ObjectiveMinimumTrace, 'r', 'LineWidth', 2)
plot(iteration(idx_min), results.ObjectiveMinimumTrace(idx_min), 'MarkerEdgeColor','black',...
    'MarkerFaceColor','gree', 'Marker', 'square', 'MarkerSize',10);
h=xlabel('Iteration index $i$ (-)');
set(h,'Interpreter', 'Latex');
h=ylabel('Performance cost  $\tilde J$ (-)');
set(h,'Interpreter', 'Latex');
legend('Current point', 'Current best point', 'Overall best point');
grid('on');
% position and size
x0=10;
y0=10;
width=450;
height=350*0.9;
set(f,'position',[x0,y0,width,height])
set(f, 'color', 'w');
% export_fig('iterations.pdf');

%%

clear t
t = time;

Kp = diag([results.bestPoint.Kp1, results.bestPoint.Kp2, results.bestPoint.Kp3, results.bestPoint.Kp4, results.bestPoint.Kp5, results.bestPoint.Kp6, results.bestPoint.Kp7]);
Ki = diag([results.bestPoint.Ki1, results.bestPoint.Ki2, results.bestPoint.Ki3, results.bestPoint.Ki4, results.bestPoint.Ki5, results.bestPoint.Ki6, results.bestPoint.Ki7]);
Kd = diag([results.bestPoint.Kd1, results.bestPoint.Kd2, results.bestPoint.Kd3, results.bestPoint.Kd4, results.bestPoint.Kd5, results.bestPoint.Kd6, results.bestPoint.Kd7]);

B               = zeros(n_DoFs,n_DoFs,length(t));
g               = zeros(length(t),n_DoFs);
tau_l           = zeros(length(t),n_DoFs);
q_msr           = zeros(length(t),n_DoFs);
dq_msr          = zeros(length(t),n_DoFs);
ddq_msr         = zeros(length(t),n_DoFs);
tau_PID         = zeros(length(t),n_DoFs);
tau_comp        = zeros(length(t),n_DoFs);

q_msr(1,:) = q_r(1,:);

B(:,:,1)  = Robot.inertia(q_r(1,:));
g(1,:)    = Robot.gravload(q_r(1,:));

tau_l(1,:) = g(1,:)' + friction*dq_msr(1,:)';

q_err = 0;

wb = waitbar(0,'Please wait...');

ierr_m(1,:) = [0 0 0 0 0 0 0];

for jj=2:length(t)

    B(:,:,jj)  = Robot.inertia(q_msr(jj-1,:));
    g(jj,:)    = Robot.gravload(q_msr(jj-1,:));

    tau_l(jj,:)    = g(jj,:)' + friction*dq_msr(jj,:)';
    tau_PID(jj,:)  = B(:,:,jj)*(Kp * (q_r(jj,:) - q_msr(jj-1,:))' - Kd * dq_msr(jj-1,:)' + Ki * ierr_m(jj-1,:)');
    tau_comp(jj,:) = g(jj,:)' + friction*dq_msr(jj,:)';

    Beq = B(:,:,jj);

    ddq_msr(jj,:) = (Beq)\(-tau_l(jj,:)' + tau_PID(jj,:)' + tau_comp(jj,:)');
    dq_msr(jj,:) = dq_msr(jj-1,:) + ddq_msr(jj,:)*Ts;
    q_msr(jj,:) = q_msr(jj-1,:) + dq_msr(jj,:)*Ts;

    ierr_m(jj,:) = ierr_m(jj-1,:) + (q_r(jj,:) - q_msr(jj,:)) * Ts;

    q_err = q_err + abs(q_r(jj,:) - q_msr(jj,:));

    waitbar(jj/length(t),wb);

end

J_err = sum(q_err)/length(t);

close(wb)


step = 100;

figure
for jj=1:step:length(t)
    plot(Robot,q_msr(jj,:));
end

figure
plot(t,(q_r-q_msr)*180/pi);
xlabel('time [s]');
ylabel('[degree]');
legend('eq1','eq2','eq3','eq4','eq5','eq6','eq7')
grid

figure
plot(t,q_r);
hold on
plot(t,q_msr);
xlabel('time [s]');
ylabel('[rad]');
legend('q1','q2','q3','q4','q5','q6','q7','qmsr1','qmsr2','qmsr3','qmsr4','qmsr5','qmsr6','qmsr7')
grid

figure
plot(t,dq_r);
hold on
plot(t,dq_msr);
xlabel('time [s]');
ylabel('dq [rad/s]');
legend('dq1','dq2','dq3','dq4','dq5','dq6','dq7','dqmsr1','dqmsr2','dqmsr3','dqmsr4','dqmsr5','dqmsr6','dqmsr7')
grid

%%

OBJ_VEC = [OBJ_VEC; obj_val];

save results_panda_PID results
