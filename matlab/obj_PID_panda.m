function J = obj_PID_panda(x_var, const, flag)

    if nargin < 3
        flag = false;
    end
    %% Count the number of function evaluations %%
    persistent idx_sim;
    if isempty(idx_sim)
        idx_sim = 1;
    end
    %% assign all constant values %%
    
    sim_var = const;
    % assign bayesian optimization variables (overwrite default constants
    % if there exist a field with the same name )
    assert(size(x_var,1) == 1)
    x_var_struct = table2struct(x_var(1,:));
    fname = fieldnames(x_var_struct);
    for idx_fn = 1:length(fname)
       sim_var.(fname{idx_fn}) = x_var_struct.(fname{idx_fn});
    end
    
    %% Simulation settings - inner loop and outer loop configuration %%
    %% 
    % Setup PID object (inner loop controller)
    Kp = diag([sim_var.Kp1, sim_var.Kp2, sim_var.Kp3, sim_var.Kp4, sim_var.Kp5, sim_var.Kp6, sim_var.Kp7]);
    Ki = diag([sim_var.Ki1, sim_var.Ki2, sim_var.Ki3, sim_var.Ki4, sim_var.Ki5, sim_var.Ki6, sim_var.Ki7]);
    Kd = diag([sim_var.Kd1, sim_var.Kd2, sim_var.Kd3, sim_var.Kd4, sim_var.Kd5, sim_var.Kd6, sim_var.Kd7,]);
    
    Robot_eval = panda_robot();
    
    %%
    
    toll_qerr = const.toll_qerr;
    Ts = const.Ts;
    t = const.time;
    n_DoFs = const.n_DoFs;
    Robot = const.Robot;
    q_r = const.r.q_r;
    dq_r = const.r.dq_r;
    ddq_r = const.r.ddq_r;
    f = const.Robot_friction;
        
    %%
    % Run the simulation
    
    g_eval = zeros(length(t),n_DoFs);
    B_eval = zeros(n_DoFs,n_DoFs,length(t));
    
    B            = zeros(n_DoFs,n_DoFs,length(t));
    g            = zeros(length(t),n_DoFs);
    tau_l        = zeros(length(t),n_DoFs);
    q_msr        = zeros(length(t),n_DoFs);
    qerr         = zeros(length(t),n_DoFs);
    dqerr        = zeros(length(t),n_DoFs);
    ddqerr       = zeros(length(t),n_DoFs);
    dq_msr       = zeros(length(t),n_DoFs);
    ddq_msr      = zeros(length(t),n_DoFs);
    tau_PID      = zeros(length(t),n_DoFs);
    tau_comp     = zeros(length(t),n_DoFs);
    
    q_msr(1,:) = q_r(1,:);
    
    B(:,:,1)  = Robot.inertia(q_r(1,:));
    g(1,:)    = Robot.gravload(q_r(1,:));
    
    B_eval(:,:,1) = Robot_eval.inertia(q_r(1,:));
    g_eval(1,:)   = Robot_eval.gravload(q_r(1,:));
    
    wb = waitbar(0,'Please wait...');
    
    ierr_m(1,:) = [0 0 0 0 0 0 0];
    
    jj = 1;
    exit_flag = false;

    penalty = [0 0 0 0 0 0 0];
    
    length(t)
    
    while ( jj<length(t) && exit_flag==false)
        
        jj=jj+1;
       
        B(:,:,jj)  = Robot.inertia(q_msr(jj-1,:));
        g(jj,:)    = Robot.gravload(q_msr(jj-1,:));
        
        B_eval(:,:,jj) = Robot_eval.inertia(q_msr(jj-1,:)); % q_msr or q_r
        g_eval(jj,:)   = Robot_eval.gravload(q_msr(jj-1,:)); % q_msr or q_r
        
        tau_l(jj,:)    = f*dq_msr(jj-1,:)' + g(jj,:)' ;
        tau_PID(jj,:)  = B_eval(:,:,jj)*(Kp * (q_r(jj,:) - q_msr(jj-1,:))' - Kd * dq_msr(jj-1,:)' + Ki * ierr_m(jj-1,:)'); % B_eval(:,:,jj) * (ddq_r(jj,:)'
        tau_comp(jj,:) = f*dq_msr(jj-1,:)' + g_eval(jj,:)';
        
        Beq = B(:,:,jj);
        
        ddq_msr(jj,:) = (Beq)\( - tau_l(jj,:)' + tau_PID(jj,:)' + tau_comp(jj,:)');
        dq_msr(jj,:) = dq_msr(jj-1,:) + ddq_msr(jj,:)*Ts;
        q_msr(jj,:) = q_msr(jj-1,:) + dq_msr(jj,:)*Ts;
        
        ierr_m(jj,:) = ierr_m(jj-1,:) + (q_r(jj,:) - q_msr(jj,:)) * Ts;
        
        qerr(jj,:)   = q_r(jj,:) - q_msr(jj,:);
        dqerr(jj,:)  = dq_r(jj,:) - dq_msr(jj,:);
        ddqerr(jj,:) = ddq_r(jj,:) - ddq_msr(jj,:);
        
        for kk=1:n_DoFs
            
            if (isnan(q_msr(jj,kk))==true)
                exit_flag = true;
                disp('jj');
                disp(jj);
                disp('exiting from optimization because nan');
                penalty(kk) = 10^10*exp(-t(jj))+10^5*exp(-t(jj));
                break;
            elseif (abs(q_r(jj,kk)-q_msr(jj,kk))>=toll_qerr)
                exit_flag = true;
                disp('jj');
                disp(jj);
                disp('q_r(jj,kk)');
                disp(q_r(jj,kk));
                disp('q_msr(jj,kk)');
                disp(q_msr(jj,kk));
                disp('exiting from optimization because max error exceeded');
                penalty(kk) = 10^5*exp(-t(jj));
                break;
            end
            
        end
        
        waitbar(jj/length(t),wb);
        
    end
    
    if jj==length(t)
        max_err_v = sum(max(abs(dqerr)));
        max_err = sum(max(abs(qerr)));
        steady_state_dqerr = sum(abs(mean(dqerr(end-50:end,:))));
        steady_state_qerr = sum(abs(mean(qerr(end-50:end,:))));
        
        J = sum(penalty) + sum(rms(qerr)) + sum(rms(dqerr)) + sum(std(qerr)) + sum(std(dqerr)) + max_err + steady_state_qerr + max_err_v;
        % + sum(rms(ddqerr)) + sum(std(ddqerr))
    else
        steady_state_qerr = 0;
        max_err = 0;
        max_err_v = 0;
        steady_state_dqerr = 0;
        
        J = sum(penalty);
    end

    if flag
        J = [J, q_msr];
    end
    
    close(wb)
    
    disp('penalty: ')
    disp(sum(penalty))
    disp('rms: ')
    disp(sum(rms(qerr)))
    disp('std: ')
    disp(sum(std(qerr)))
    disp('max err: ')
    disp(max_err)
    disp('steady state qerr: ')
    disp(steady_state_qerr)
    disp('max err vel: ')
    disp(max_err_v)
    disp('steady state dqerr: ')
    disp(steady_state_dqerr)
    disp('std vel: ')
    disp(sum(std(dqerr)))
    disp('rms vel: ')
    disp(sum(rms(dqerr)))
    fprintf('Function evaluation %.0f: final cost: %12.8f \n', idx_sim, J)
    fprintf('--------------------------------\n')
    idx_sim = idx_sim + 1;
    
end
