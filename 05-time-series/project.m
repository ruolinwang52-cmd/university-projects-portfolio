
%% Part A, Section 1: Model Selection Using ARMA
clear all;
clc;
close all;

%% A.1.1 Load the data and plot temperature vs hours

load('projectData24.mat'); 
temperature = data(:, 8); 
year = data(:, 1);         
month = data(:, 2);       
day = data(:, 3);         
hour = data(:, 4);

% Create a datetime array for proper time indexing
date_time = datetime(year, month, day, hour, 0, 0);
figure;
plot(date_time, temperature, 'b', 'LineWidth', 1.5);
title('Air Temperature vs Time (Hours)');
xlabel('Time');
ylabel('Air Temperature (°C)');
grid on;
%% A.1.2 Box-Cox Transformation
% Determine if a transformation is needed using the Box-Cox curve
figure;
lambda_max = bcNormPlot(temperature, 1); 
fprintf('The Box-Cox curve is maximized at %4.2f. This means no transformation applied since lambda_max =0.95 is close to 1.\n', lambda_max);

% Note: No transformation applied since lambda_max =0.95 is close to 1.

%% A.1.3 Find a Relatively Stationary 8-Week Period
% Do a rolling window of 8 weeks
rolling_window = 8 * 24 * 7; 
rolling_std = movstd(temperature, rolling_window);

% Adjust the time vector to match the rolling standard deviation length
adjusted_time = date_time(rolling_window:end);
figure;
plot(adjusted_time, rolling_std(rolling_window:end), 'r', 'LineWidth', 1.5);
title('Rolling Standard Deviation of Raw Data (2-Week Window)');
xlabel('Time');
ylabel('Rolling Std Dev');
grid on;

% Define a threshold for stability
threshold = 2.21; 

% Find indices where rolling standard deviation is below the threshold
stable_indices = find(rolling_std(rolling_window:end) < threshold);

if ~isempty(stable_indices)
 
    start_index = stable_indices(1) + rolling_window - 1; % Adjust for rolling window offset
    end_index = min(start_index + rolling_window - 1, length(temperature));
    stable_data = temperature(start_index:end_index);
    stable_days = date_time(start_index:end_index);

    figure;
    plot(stable_days, stable_data, 'g', 'LineWidth', 1.5);
    title('Stable 8-Week Interval of Raw Data');
    xlabel('Time');
    ylabel('Air Temperature (°C)');
    grid on;

    % Display the start and end of the stable interval
    fprintf('Stable interval found from %s to %s.\n', datestr(stable_days(1)), datestr(stable_days(end)));
else
    disp('No stable interval found within the specified threshold.');
end


stable_data=temperature(start_index:end_index);
stable_days = date_time(start_index:end_index);



%% A.1.3.1 choosing validation data and test data
%Validation data is choosen as theweek directly after model data
validation_data = temperature(end_index+1:end_index + 24*7 );
validation_days = date_time(end_index+1:end_index+ 24*7 );
validation_start=end_index+1;
validation_stop=end_index+ 24*7;
%test data is chosen as the 4 weeks directly after validation data
Test_length= 24*7*4;
test1_data= temperature(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
test1_days = date_time(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
test1_start=end_index+ 24*7+1;
test1_stop=end_index+ 24*7+Test_length;
%Additional Sections of test data will/can be added such as same period
%next year or summer and spring.
%Test data of same period the year after
test2_data= temperature(start_index+ 365*24 :start_index+ 365*24+Test_length);
test2_days = date_time(start_index+ 365*24 :start_index+ 365*24+Test_length);
test2_start=start_index+ 365*24 ;
test2_stop = start_index+ 365*24+Test_length;
%Test data of following summer, it is expected that model will be worst for
%summer since model is based on a period in the winter.
test3_data= temperature(start_index+ 140*24 :start_index+ 140*24+Test_length);
test3_days = date_time(start_index+ 140*24 :start_index+ 140*24+Test_length);
test3_start =start_index+ 140*24;
test3_stop =start_index+ 140*24+Test_length;
%% A.1.3.2 plotting of validation and test data
figure;
plot(validation_days,validation_data)
title('validation data')
xlabel('Time');
ylabel('Air Temperature (°C)');
figure;
plot(test1_days,test1_data)
title('test data 1')
xlabel('Time');
ylabel('Air Temperature (°C)');
figure;
plot(test2_days,test2_data)
title('test data 2')
xlabel('Time');
ylabel('Air Temperature (°C)');
figure;
plot(test3_days,test3_data)
title('test data 3')
xlabel('Time');
ylabel('Air Temperature (°C)');

figure;
subplot(4, 1, 1);
plot(validation_days,validation_data)
title('validation data')
xlabel('Time');
ylabel('Air Temperature (°C)');
subplot(4, 1, 2);
plot(test1_days,test1_data)
title('test data 1')
xlabel('Time');
ylabel('Air Temperature (°C)');
subplot(4, 1, 3);
plot(test2_days,test2_data)
title('test data 2')
xlabel('Time');
ylabel('Air Temperature (°C)');
subplot(4,1,4)
plot(test3_days,test3_data)
title('test data 3')
xlabel('Time');
ylabel('Air Temperature (°C)');
%% A.1.4 Plot ACF and PACF for the Stable 8-Week Interval
if ~isempty(stable_indices)
    noLags = 50; 
    % Compute and plot ACF and PACF for the stable data
    [rhoEst, phiEst] = plotACFnPACF(stable_data, noLags, ...
        'ACF and PACF of Stable 8-Week Interval');
else
    disp('No stable interval available for ACF and PACF analysis.');
end

% From ACF it is decaying slowly which suggest we should differentiate
% data

%% A.1.5 Apply First-Order Differentiation to Stable Data
% Do first-order differentiation polynomial
AS = [1 -1]; 
stable_data_diff1 = filter(AS, 1, stable_data); 
stable_data_diff1 = stable_data_diff1(length(AS):end); % Remove initial samples

% Adjust the time vector for the first differentiation
stable_days_diff1 = stable_days(length(AS):end);
figure;
plot(stable_days_diff1, stable_data_diff1, 'b', 'LineWidth', 1.5);
title('First-Order Differentiated Stable Data');
xlabel('Time');
ylabel('Differentiated Temperature(°C)');
grid on;

%% A.1.6 Plot ACF and PACF for First-Order Differentiated Stable Data
noLags_diff1 = 30; 
[rhoEst_diff1, phiEst_diff1] = plotACFnPACF(stable_data_diff1, noLags_diff1, ...
    'ACF and PACF of First-Order Differentiated Stable Data');

% From ACF, it is clear that there is a season of 24, so we remove the
% season
%% A.1.7 Apply Seasonal Differentiation (24-Hour Periodicity)
% Do the seasonal differentiation 
sseason_24 = 24; 
seasonPoly_24 = [1 zeros(1, sseason_24-1) -1]; 
stable_data_diff_season_24 = filter(seasonPoly_24, 1, stable_data_diff1); 
stable_data_diff_season_24 = stable_data_diff_season_24(length(seasonPoly_24):end); % Remove initial samples

% Adjust the time vector for the seasonal differentiation
stable_days_diff_season_24 = stable_days_diff1(length(seasonPoly_24):end);
figure;
plot(stable_days_diff_season_24, stable_data_diff_season_24, 'b', 'LineWidth', 1.5);
title('Seasonally Differentiated Stable Data (24-Hour Periodicity)');
xlabel('Time');
ylabel('Seasonally Differentiated Temperature(°C)');
grid on;

%% A.1.7.1 Plot ACF and PACF for Seasonally Differentiated Stable Data
noLags_season_24 = 25; 
[rhoEst_season_24, phiEst_season_24] = plotACFnPACF(stable_data_diff_season_24, noLags_season_24, ...
    'ACF and PACF of Seasonally Differentiated Stable Data');

%From ACF and PACF, we try AR3 with ar2 coef =0 

%% A.1.8 Estimate ARMA Model for Stable Data
% Test AR(1)  
noLags_stable = 25; 
arma_model = estimateARMA(stable_data_diff_season_24, [1 1 ], [1], ...
    'ARMA Model for Seasonally Differentiated Stable Data', noLags_stable);
disp('Estimated ARMA Model Parameters:');
present(arma_model);

% All coef are significant. 
%% A.1.8.1 Increasing order of MA part
%test ARMA(1,24) instead
noLags_stable = 25; 
[arma_model, pacfEst] = estimateARMA(stable_data_diff_season_24, [1 1], [1 zeros(1,23) 1], ...
    'ARMA Model for Seasonally Differentiated Stable Data', noLags_stable);
disp('Estimated ARMA Model Parameters:');
present(arma_model);

%% A.1.9 Can we trust the Monti test?
checkIfNormal( pacfEst(2:end), 'PACF' );

%  PACF is NOT normal distributed.
%hence we can trust it less.
%% Part A, Section 2: Predict Data on Validation Period and Test period 
Amodel_A= arma_model.A;
Amodel_B= arma_model.C;
firstdiff= AS;
seconddiff=seasonPoly_24;
differention = conv(firstdiff,seconddiff);
Amodel_A= conv(differention,Amodel_A);
reslag=30;
%% A.2.1 step prediction of entire temperature data
%1-step prediction
[original_temperature, temp_modelA_k1,modelA_residue_k1] =predictARMA(temperature,Amodel_A,Amodel_B, 1,1);
%3-step prediction
[original_temperature,temp_modelA_k3, modelA_residue_k3] =predictARMA(temperature,Amodel_A,Amodel_B, 3,1);
%4-step prediction
[original_temperature,temp_modelA_k4, modelA_residue_k4] =predictARMA(temperature,Amodel_A,Amodel_B, 4,1);
%18-step prediction
[original_temperature,temp_modelA_k18, modelA_residue_k18] =predictARMA(temperature,Amodel_A,Amodel_B, 18,1);
%The naive predictor is chosen as the temperature 24 hours before
naive=temperature;
naive(1:24)=zeros(24,1);
naive(25:end)=temperature(1:length(temperature)-24);
naive_res=temperature-naive;

%% A.2.1.1 plotting the entire period
figure;
plot(date_time,[temperature,temp_modelA_k1,temp_modelA_k3,temp_modelA_k4,temp_modelA_k18, naive])
title('Model A prediction of temperature')
legend('temperature', 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(date_time,[modelA_residue_k1,modelA_residue_k3,modelA_residue_k4,modelA_residue_k18,naive_res])
title('Model A prediction residual of temperature')
legend( 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Temperature error (°C)');
%% A.2.1 extracting predictions of validation
figure;
plot(validation_days,[validation_data,temp_modelA_k1(validation_start:validation_stop),temp_modelA_k3(validation_start:validation_stop),temp_modelA_k4(validation_start:validation_stop),temp_modelA_k18(validation_start:validation_stop),naive(validation_start:validation_stop)])
title('Model A predictions of validation data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(validation_days,[modelA_residue_k1(validation_start:validation_stop),modelA_residue_k3(validation_start:validation_stop),modelA_residue_k4(validation_start:validation_stop),modelA_residue_k18(validation_start:validation_stop),naive_res(validation_start:validation_stop)])
title('Residues of Model A predictions of validation data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(validation_start:validation_stop),reslag,'model A validation residue k=1');
plotACFnPACF(modelA_residue_k3(validation_start:validation_stop),reslag,'model A validation residue k=3');
plotACFnPACF(modelA_residue_k4(validation_start:validation_stop),reslag,'model A validation residue k=4');
plotACFnPACF(modelA_residue_k18(validation_start:validation_stop),reslag,'model A validation residue k=18');

var_valid=var(validation_data);
fprintf('\n Prediction results of validation for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
tilte()
whitenessTest(modelA_residue_k3(validation_start:validation_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(modelA_residue_k4(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the validations data is    %7.2f\n', var(normalize(validation_data )))
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k4(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(validation_start:validation_stop))/var_valid)*100 )
%% A.2.1(b) extracting predictions of validation without k=4
figure;
plot(validation_days,validation_data,'k','Linewidth',1.5)
hold on
plot(validation_days,[temp_modelA_k1(validation_start:validation_stop), ...
    temp_modelA_k3(validation_start:validation_stop), ...
    temp_modelA_k18(validation_start:validation_stop),naive(validation_start:validation_stop)])
title('Model A predictions of validation data')
legend('temperature','k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
hold off
plot(validation_days,[modelA_residue_k1(validation_start:validation_stop), ...
    modelA_residue_k3(validation_start:validation_stop), ...
    modelA_residue_k18(validation_start:validation_stop),naive_res(validation_start:validation_stop)])
title('Residues of Model A predictions of validation data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(validation_start:validation_stop),reslag,'model A validation residue k=1');
plotACFnPACF(modelA_residue_k3(validation_start:validation_stop),reslag,'model A validation residue k=3');
plotACFnPACF(modelA_residue_k18(validation_start:validation_stop),reslag,'model A validation residue k=18');

var_valid=var(validation_data);
fprintf('\n Prediction results of validation for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the validations data is    %7.2f\n', var(normalize(validation_data )))
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(validation_start:validation_stop))/var_valid)*100 )
%% A.2.3 Extracting prediction of test data
%% A.2.3.1 test 1
figure;
plot(test1_days,[test1_data,temp_modelA_k1(test1_start:test1_stop),temp_modelA_k3(test1_start:test1_stop),temp_modelA_k4(test1_start:test1_stop),temp_modelA_k18(test1_start:test1_stop),naive(test1_start:test1_stop)])
title('Model A predictions of test1 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test1_days,[modelA_residue_k1(test1_start:test1_stop),modelA_residue_k3(test1_start:test1_stop),modelA_residue_k4(test1_start:test1_stop),modelA_residue_k18(test1_start:test1_stop),naive_res(test1_start:test1_stop)])
title('Residues of Model A predictions of  test1 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error (°C)');
hold off
plotACFnPACF(modelA_residue_k1(test1_start:test1_stop),reslag,'model A test1 residue k=1')
plotACFnPACF(modelA_residue_k3(test1_start:test1_stop),reslag,'model A test1 residue k=3')
plotACFnPACF(modelA_residue_k4(test1_start:test1_stop),reslag,'model A test1 residue k=4')
plotACFnPACF(modelA_residue_k18(test1_start:test1_stop),reslag,'model A test1 residue k=18')

var_test1=var(test1_data);
fprintf('\n Prediction results of test1 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test1_start:test1_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(modelA_residue_k4(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the test1 data is    %7.2f\n', var_test1 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop))/var_test1 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k4(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test1_start:test1_stop))/var_test1)*100 )
%% A.2.3.1(b) test 1 without k=4
figure;
plot(test1_days,test1_data,'k','Linewidth',1.5)
hold on
plot(test1_days,[temp_modelA_k1(test1_start:test1_stop),temp_modelA_k3(test1_start:test1_stop),temp_modelA_k18(test1_start:test1_stop),naive(test1_start:test1_stop)])
title('Model A predictions of test1 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
hold off
figure;
plot(test1_days,[modelA_residue_k1(test1_start:test1_stop),modelA_residue_k3(test1_start:test1_stop),modelA_residue_k18(test1_start:test1_stop),naive_res(test1_start:test1_stop)])
title('Residues of Model A predictions of  test1 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error (°C)');
hold off
plotACFnPACF(modelA_residue_k1(test1_start:test1_stop),reslag,'model A test1 residue k=1')
plotACFnPACF(modelA_residue_k3(test1_start:test1_stop),reslag,'model A test1 residue k=3')
plotACFnPACF(modelA_residue_k18(test1_start:test1_stop),reslag,'model A test1 residue k=18')

var_test1=var(test1_data);
fprintf('\n Prediction results of test1 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the test1 data is    %7.2f\n', var_test1 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop))/var_test1 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test1_start:test1_stop))/var_test1)*100 )
%% A.2.3.3  test 2
figure;
plot(test2_days,[test2_data,temp_modelA_k1(test2_start:test2_stop),temp_modelA_k3(test2_start:test2_stop),temp_modelA_k4(test2_start:test2_stop),temp_modelA_k18(test2_start:test2_stop),naive(test2_start:test2_stop)])
title('Model A predictions of test2 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test2_days,[modelA_residue_k1(test2_start:test2_stop),modelA_residue_k3(test2_start:test2_stop),modelA_residue_k4(test2_start:test2_stop),modelA_residue_k18(test2_start:test2_stop),naive_res(test2_start:test2_stop)])
title('Residues of Model A predictions of  test2 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(test2_start:test2_stop),reslag,'model A test2 residue k=1')
plotACFnPACF(modelA_residue_k3(test2_start:test2_stop),reslag,'model A test2 residue k=3')
plotACFnPACF(modelA_residue_k4(test2_start:test2_stop),reslag,'model A test2 residue k=4')
plotACFnPACF(modelA_residue_k18(test2_start:test2_stop),reslag,'model A test2 residue k=18')

var_test2=var(test2_data);
fprintf('\n Prediction results of test2 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test2_start:test2_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(modelA_residue_k4(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test2 data is    %7.2f\n', var_test2 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop))/var_test2 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k4(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test2_start:test2_stop))/var_test2)*100 )
%% A.2.3.3(b)  test 2 without k=4
figure;

plot(test2_days,test2_data,'k','Linewidth',1.5)
hold on
plot(test2_days,[temp_modelA_k1(test2_start:test2_stop),temp_modelA_k3(test2_start:test2_stop),temp_modelA_k18(test2_start:test2_stop),naive(test2_start:test2_stop)])
title('Model A predictions of test2 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
hold off
figure;
plot(test2_days,[modelA_residue_k1(test2_start:test2_stop),modelA_residue_k3(test2_start:test2_stop),modelA_residue_k18(test2_start:test2_stop),naive_res(test2_start:test2_stop)])
title('Residues of Model A predictions of  test2 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(test2_start:test2_stop),reslag,'model A test2 residue k=1')
plotACFnPACF(modelA_residue_k3(test2_start:test2_stop),reslag,'model A test2 residue k=3')
plotACFnPACF(modelA_residue_k18(test2_start:test2_stop),reslag,'model A test2 residue k=18')

var_test2=var(test2_data);
fprintf('\n Prediction results of test2 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test2 data is    %7.2f\n', var_test2 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop))/var_test2 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test2_start:test2_stop))/var_test2)*100 )
%% A.2.3.3 test 3
figure;
plot(test3_days,[test3_data,temp_modelA_k1(test3_start:test3_stop),temp_modelA_k3(test3_start:test3_stop),temp_modelA_k4(test3_start:test3_stop),temp_modelA_k18(test3_start:test3_stop),naive(test3_start:test3_stop)])
title('Model A predictions of test3 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test3_days,[modelA_residue_k1(test3_start:test3_stop),modelA_residue_k3(test3_start:test3_stop),modelA_residue_k4(test3_start:test3_stop),modelA_residue_k18(test3_start:test3_stop),naive_res(test3_start:test3_stop)])
title('Residues of Model A predictions of  test3 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(test3_start:test3_stop),reslag,'model A test3 residue k=1')
plotACFnPACF(modelA_residue_k3(test3_start:test3_stop),reslag,'model A test3 residue k=3')
plotACFnPACF(modelA_residue_k4(test3_start:test3_stop),reslag,'model A test3 residue k=4')
plotACFnPACF(modelA_residue_k18(test3_start:test3_stop),reslag,'model A test3 residue k=18')

var_test3=var(test3_data);
fprintf('\n Prediction results of test3 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test3_start:test3_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test3_start:test3_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(modelA_residue_k4(test3_start:test3_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test3_start:test3_stop))
fprintf('variance:\n')
fprintf('  The variance of the test3 data is    %7.2f\n', var_test3 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(modelA_residue_k4(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop))/var_test3 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k4(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test3_start:test3_stop))/var_test3)*100 )
%% A.2.3.3(b) test 3 with 4=k
figure;
plot(test3_days,test3_data,'k','Linewidth',1.5)
hold on
plot(test3_days,[temp_modelA_k1(test3_start:test3_stop),temp_modelA_k3(test3_start:test3_stop),temp_modelA_k18(test3_start:test3_stop),naive(test3_start:test3_stop)])
title('Model A predictions of test3 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
hold off
figure;
plot(test3_days,[modelA_residue_k1(test3_start:test3_stop),modelA_residue_k3(test3_start:test3_stop),modelA_residue_k18(test3_start:test3_stop),naive_res(test3_start:test3_stop)])
title('Residues of Model A predictions of  test3 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(modelA_residue_k1(test3_start:test3_stop),reslag,'model A test3 residue k=1')
plotACFnPACF(modelA_residue_k3(test3_start:test3_stop),reslag,'model A test3 residue k=3')
plotACFnPACF(modelA_residue_k18(test3_start:test3_stop),reslag,'model A test3 residue k=18')

var_test3=var(test3_data);
fprintf('\n Prediction results of test3 for model A:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(modelA_residue_k1(test3_start:test3_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(modelA_residue_k3(test3_start:test3_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(modelA_residue_k18(test3_start:test3_stop))
fprintf('variance:\n')
fprintf('  The variance of the test3 data is    %7.2f\n', var_test3 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(modelA_residue_k1(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(modelA_residue_k3(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(modelA_residue_k18(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop))/var_test3 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k1(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k3(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(modelA_residue_k18(test3_start:test3_stop))/var_test3)*100 )
%%
%% Part B, Section 1: Model Selection Using BJ Model with Radiation (Input) vs. Temperature (Output)


%% B.1.1 Load the data and plot temperature vs hours
load('projectData24.mat'); 
radiation = data(:, 6); % Net Radiation (input)
temperature = data(:, 8); % Temperature (output)
year = data(:, 1);         
month = data(:, 2);       
day = data(:, 3);         
hour = data(:, 4);
date_time = datetime(year, month, day, hour, 0, 0);
figure;
plot(date_time, radiation, 'r', 'LineWidth', 1.5, 'DisplayName', 'Radiation (Input)');
hold on;
plot(date_time, temperature, 'b', 'LineWidth', 1.5, 'DisplayName', 'Temperature (Output)');
title('Radiation and Temperature vs. Time');
xlabel('Time');
ylabel('Amplitude');
legend('show');
grid on;

%% B.1.2 Cross-Correlation Function (CCF)
noLags = 50; 
[rhoCCF, lags] = xcorr(temperature, radiation, noLags, 'coeff'); % Cross-correlation
figure;
stem(lags, rhoCCF, 'b', 'LineWidth', 1.5);
hold on;
confInterval = 2/sqrt(length(radiation)); % Confidence interval
plot(lags, confInterval*ones(size(lags)), 'r--', 'LineWidth', 1.5);
plot(lags, -confInterval*ones(size(lags)), 'r--', 'LineWidth', 1.5);
title('Cross-Correlation Between Radiation (Input) and Temperature (Output)');
xlabel('Lag');
ylabel('Correlation');
grid on;

% As we see from plot, input and output are clearly related, so we can use
% BJ model. 

%% B.1.3 Should We Transform The Data? Check Box-Cox
figure;
subplot(1, 2, 1);
lambda_radiation = bcNormPlot(radiation, 1);
title('Box-Cox Transformation for Radiation (Input)');
xlabel('Lambda');
ylabel('Log-Likelihood');
fprintf('The Box-Cox curve for Radiation is maximized at %4.2f.\n', lambda_radiation);

subplot(1, 2, 2);
lambda_temperature = bcNormPlot(temperature, 1);
title('Box-Cox Transformation for Temperature (Output)');
xlabel('Lambda');
ylabel('Log-Likelihood');
fprintf('The Box-Cox curve for Temperature is maximized at %4.2f.\n', lambda_temperature);

% Note: Need log transformation applied since lambda_max =-0.30 is close to
% 0. But values are sometimes negative, so we do not take log here.
% Note: No transformation applied to OUTPUT since lambda_max =0.95 is close to 1.




%% B.1.4  Define Stable Period for Radiation and Temperature Again
%The same period is chosen as in Part A
stable_radiation = radiation(start_index:end_index);
stable_temperature = temperature(start_index:end_index);
stable_time = date_time(start_index:end_index);
%% B.1.4.1 Define validation and test periods for radiation

%Validation data is choosen as the week directly after model data
validation_radiation = radiation(end_index+1:end_index + 24*7 );
%test data is chosen as the 4 weeks directly after validation data
Test_length= 24*7*4;
test_radiation1= radiation(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
%Additional Sections of test data will/can be added such as same period
%next year or summer and spring.
%Test data of same period the year after
test_radiation2= radiation(start_index+ 365*24 :start_index+ 365*24+Test_length);
%Test data of following summer, it is expected that model will be worst for
%summer since model is based on a period in the winter.
test_radiation3= radiation(start_index+ 140*24 :start_index+ 140*24+Test_length);
%% B.1.5 Plot Radiation and Temperature and Cross-Correlation Function During Stable Period
figure;
plot(stable_time, stable_radiation, 'r', 'LineWidth', 1.5, 'DisplayName', 'Radiation (Input)');
hold on;
plot(stable_time, stable_temperature, 'b', 'LineWidth', 1.5, 'DisplayName', 'Temperature (Output)');
title('Radiation and Temperature During Stable Period');
xlabel('Time');
ylabel('Amplitude');
legend('show');
grid on;
noLags = 50; 
[rhoCCF_stable, lags_stable] = xcorr(stable_temperature, stable_radiation, noLags, 'coeff'); % Cross-correlation

% Plot the CCF for the stable period
figure;
stem(lags_stable, rhoCCF_stable, 'b', 'LineWidth', 1.5);
hold on;
confInterval = 2/sqrt(length(stable_radiation)); % Confidence interval
plot(lags_stable, confInterval*ones(size(lags_stable)), 'r--', 'LineWidth', 1.5);
plot(lags_stable, -confInterval*ones(size(lags_stable)), 'r--', 'LineWidth', 1.5);
title('Cross-Correlation Between Radiation (Input) and Temperature (Output) During Stable Period');
xlabel('Lag');
ylabel('Correlation');
grid on;

% From CCF, the input and output are clearly related in the 8 weeks stable period 
%% B.1.6 ACF PACF for Input (Radiation)

noLags_input = 50; 
% Plot ACF and PACF for the input data (Radiation)
[rhoEst_radiation, phiEst_radiation] = plotACFnPACF(stable_radiation, noLags_input, ...
    'ACF and PACF of Radiation (Input) During Stable Period');

% ACF of input has a periodicity of 24 hours 


%% B.1.9 Apply Seasonal Differentiation (24-Hour Periodicity)
% Apply the seasonal differentiation polynomial for 24-hour periodicity
season_lag = 24; 
seasonal_diff_poly = [1 zeros(1, season_lag-1) -1]; 

% Apply seasonal differencing directly to the stable radiation
radiation_seasonal_diff = filter(seasonal_diff_poly, 1, stable_radiation); 
radiation_seasonal_diff = radiation_seasonal_diff(season_lag+1:end); % Remove initial corrupted samples

% Adjust the time vector for the seasonally differenced data
radiation_seasonal_diff_time = stable_time(season_lag+1:end);

% Plot the seasonally differenced input data
figure;
plot(radiation_seasonal_diff_time, radiation_seasonal_diff, 'r', 'LineWidth', 1.5);
title('Seasonally Differenced Radiation (Input, 24-Hour Periodicity)');
xlabel('Time');
ylabel('Seasonally Differenced Net Radiation(W/m^2)');
grid on;

%% B.1.10 ACF and PACF for Seasonally Differenced Radiation
% Compute and plot ACF and PACF for the seasonally differenced input data
[rhoEst_diff_season_radiation, phiEst_diff_season_radiation] = plotACFnPACF(radiation_seasonal_diff, noLags_input, ...
    'ACF and PACF of Seasonally Differenced Radiation (Input)');


% Since we removed season 24, so we must add ma24, from pacf we need to add
% ar1
%% B.1.11 Estimate ARMA(3,24) Model for Seasonally Differenced Radiation

AR_order = [1 1 1]; % AR(3)
MA_order = [1 zeros(1, 23) 1]; % MA(24), only the 24th coefficient

% Estimate the ARMA model
noLags_model = 25; 
[arma_model_radiation , pacfEstinput]= estimateARMA(radiation_seasonal_diff, AR_order, MA_order, ...
    'ARMA(1,24) Model for Seasonally Differenced Radiation', noLags_model);
disp('Estimated ARMA(3,24) Model Parameters:');
present(arma_model_radiation);
%% B.1.11(b) Can we trust the Monti test?
checkIfNormal( pacfEstinput(2:end), 'PACF' );

%  PACF is NOT normal distributed. We cannot trust whitenesstest
%  completely.

%% B.1.12 Pre-whitening Input and Output Signals
% Assign the ARMA model identified for the input (radiation)
foundModel = arma_model_radiation;

% Pre-whiten the input signal (Radiation)
ex = filter(foundModel.A, foundModel.C, stable_radiation);
ex = ex(length(foundModel.A+10):end); % Remove initial corrupted samples

% Pre-whiten the original output signal (Temperature)
ey = filter(foundModel.A, foundModel.C, stable_temperature);
ey = ey(length(foundModel.A+10):end); % Remove initial corrupted samples

% Plot the pre-whitened signals
figure;
subplot(2, 1, 1);
plot(ex, 'r', 'LineWidth', 1.5);
title('Pre-whitened Radiation Signal (ex)');
xlabel('Sample Index');
ylabel('Amplitude(W/m^2)');
grid on;

subplot(2, 1, 2);
plot(ey, 'b', 'LineWidth', 1.5);
title('Pre-whitened Temperature Signal (ey)');
xlabel('Sample Index');
ylabel('Amplitude(°C)');
grid on;
%% B.1.13 Cross-Correlation of Pre-whitened Signals 
% Ensure signals have the same length
minLen = min(length(ex), length(ey));
ex = ex(1:minLen);
ey = ey(1:minLen);

% Compute cross-correlation between pre-whitened input and output signals
M = 30; 
[Cxy, lags] = xcorr(ey, ex, M, 'coeff');

% Plot the cross-correlation
figure;
stem(-M:M, Cxy, 'b', 'LineWidth', 1.5);
hold on;

% Add confidence intervals
confInt = 2 / sqrt(length(ey)); % 2/sqrt(N)
plot(-M:M, confInt * ones(1, 2 * M + 1), 'r--', 'LineWidth', 1.5);
plot(-M:M, -confInt * ones(1, 2 * M + 1), 'r--', 'LineWidth', 1.5);

% Finalize the plot
hold off;
xlabel('Lag');
ylabel('Cross-Correlation');
title('Cross-Correlation Between Pre-whitened Signals');
grid on;

% Analyze the cross-correlation:
% - Delay (d): Estimated from the first significant positive lag, likely d=0. 

% - B(z) (s): Estimated based on ringing behavior starting at d+s=0, s=1,
% but s=1 is not significant, so we chose s=0
% - A2(z) (r): Inferred from decaying behavior in cross-correlation, can be
% r=0, 1, or 2. Let us try r=1 first. Doesnt work, so we switch back to r=0

%% B.1.13 finding first BJ model 
foundModelBJ= estimateBJ(stable_temperature, stable_radiation, [1], [1  ], [1 ], [1 ], 'BJ2 Model with d=0, r=0, s=0', noLags);
% From pacf seems like we need to add a1 a2

%% B.1.13(b) finding second BJ model 
foundModelBJ= estimateBJ(stable_temperature, stable_radiation, [1 ], [1 1 1 ], [ 1 ], [1 ], 'BJ2 Model with d=0, r=0, s=0', noLags);

%% Obs do not run twice
radiation = data(:, 6);
radiation = radiation;
stable_radiation=radiation(start_index:end_index);
%Validation data is choosen as the week directly after model data
validation_radiation = radiation(end_index+1:end_index + 24*7 );
%test data is chosen as the 4 weeks directly after validation data
Test_length= 24*7*4;
test_radiation1= radiation(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
%Additional Sections of test data will/can be added such as same period
%next year or summer and spring.
%Test data of same period the year after
test_radiation2= radiation(start_index+ 365*24 :start_index+ 365*24+Test_length);
%Test data of following summer, it is expected that model will be worst for
%summer since model is based on a period in the winter.
test_radiation3= radiation(start_index+ 140*24 :start_index+ 140*24+Test_length);

%% B.1.14 using pam to estimate all polynomials and analysis the residue 
A1=[1 1 1];
A2=[1 ];
B=[1];
C=[1 ];
Mi=idpoly(1,B,C,A1,A2);
z=iddata(stable_temperature,stable_radiation);
MboxJ=pem(z,Mi);
ehat=resid(MboxJ,z);
present(MboxJ);
figure;
plot(ehat.y);
title('Residuals (ehat) of the BJ Model', 'Interpreter', 'latex');
xlabel('Sample Index', 'Interpreter', 'latex');
ylabel('Amplitude', 'Interpreter', 'latex');
grid on;

plotACFnPACF(ehat.y, 24,'Acf and Pacf of residue ehat');
checkIfWhite(ehat.y);
%% Part B, Section 2: Prediction of validation and test
%% B.2.1 Defining model polynomials
radiation_seasonal_diff = filter(seasonal_diff_poly, 1, stable_radiation);
[arma_model_radiation , pacfEstinput]= estimateARMA(radiation_seasonal_diff, AR_order, MA_order, ...
    'ARMA(1,24) Model for Seasonally Differenced Radiation', noLags_model);
%we first define model polynimals for prediction
Bmodel_A=foundModelBJ.A;
Bmodel_B=foundModelBJ.B; 
Bmodel_C=foundModelBJ.C;
Bmodel_A1=foundModelBJ.D;
Bmodel_A2=foundModelBJ.F;

radiationmodel_A = arma_model_radiation.A;
radiationmodel_C = arma_model_radiation.C;
differention = seasonal_diff_poly ;
radiationmodel_A= conv(differention,radiationmodel_A);
%Intresting to note is that multiplying A,C,A1 by a factor has a negligent effect on
%prediction.
%% genualiy need to test this without functions to see if that fixes it
%but first we get result with function

%% B.2.2 prediction for entire  temperature data

%1-step prediction
[original_temperature,original_radiation,temp_modelB_k1,rad_modelB_k1,temp_modelB_res_k1,rad_modelB_res_k1]=predictBJ(temperature,radiation, Bmodel_A,Bmodel_C,Bmodel_B,Bmodel_A1,Bmodel_A2,radiationmodel_A,radiationmodel_C,1);
%3-step prediction
[original_temperature,original_radiation,temp_modelB_k3,rad_modelB_k3,temp_modelB_res_k3,rad_modelB_res_k3]=predictBJ(temperature,radiation, Bmodel_A,Bmodel_C,Bmodel_B,Bmodel_A1,Bmodel_A2,radiationmodel_A,radiationmodel_C,3);
%4-step
[original_temperature,original_radiation,temp_modelB_k4,rad_modelB_k4,temp_modelB_res_k4,rad_modelB_res_k4]=predictBJ(temperature,radiation, Bmodel_A,Bmodel_C,Bmodel_B,Bmodel_A1,Bmodel_A2,radiationmodel_A,radiationmodel_C,4);
%18-step
[original_temperature,original_radiation,temp_modelB_k18,rad_modelB_k18,temp_modelB_res_k18,rad_modelB_res_k18]=predictBJ(temperature,radiation, Bmodel_A,Bmodel_C,Bmodel_B,Bmodel_A1,Bmodel_A2,radiationmodel_A,radiationmodel_C,18);
%we also create naive predictor of radiatrion as the radiation 24 hours
%before
naive=temperature;
naive(1:24)=zeros(24,1);
naive(25:end)=temperature(1:length(temperature)-24);
naive_res=temperature-naive;

naive_rad=radiation;
naive_rad(1:24)=zeros(24,1);
naive_rad(25:end)=radiation(1:length(temperature)-24);
naive_rad_res=radiation-naive_rad;
%% B.2.2.1 plotting the entire period
%input prediction
figure;
plot(date_time,[radiation,rad_modelB_k1,rad_modelB_k3,rad_modelB_k4,rad_modelB_k18, naive_rad])
title('Model B prediction of radiation')
legend('radiation', 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(date_time,[rad_modelB_res_k1,rad_modelB_res_k3,rad_modelB_res_k4,rad_modelB_res_k4,naive_rad_res])
title('Model B prediction residual of radiation')
legend( 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Net Radiation error(W/m^2)');
%output prediction
figure;
plot(date_time,[temperature,temp_modelB_k1,temp_modelB_k3,temp_modelB_k4,temp_modelB_k18, naive])
title('Model B prediction of temperature')
legend('temperature', 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(date_time,[temp_modelB_res_k1,temp_modelB_res_k3,temp_modelB_res_k4,temp_modelB_res_k4,naive_res])
title('Model B prediction residual of temperature')
legend( 'k=1 prediction', 'k=3 prediction', 'k=4 prediction', 'k=18 prediction','naive')
xlabel('Time');
ylabel('Temperature error(°C)');

%% B.2.2 Extrating valdiation prediction
reslag=30;
%% B.2.2.1 input
figure;
plot(validation_days,[validation_radiation,rad_modelB_k1(validation_start:validation_stop),rad_modelB_k3(validation_start:validation_stop),rad_modelB_k4(validation_start:validation_stop),rad_modelB_k18(validation_start:validation_stop),naive_rad(validation_start:validation_stop)])
title('Model B predictions of radiation validation data')
legend('radiation','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(validation_days,[rad_modelB_res_k1(validation_start:validation_stop),rad_modelB_res_k3(validation_start:validation_stop),rad_modelB_res_k4(validation_start:validation_stop),rad_modelB_res_k18(validation_start:validation_stop),naive_rad_res(validation_start:validation_stop)])
title('Residues of Model B predictions predictions of validation data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radition error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(validation_start:validation_stop),reslag,'model B radiation validation residue k=1')
plotACFnPACF(rad_modelB_res_k3(validation_start:validation_stop),reslag,'model B radiation validation residue k=3')
plotACFnPACF(rad_modelB_res_k4(validation_start:validation_stop),reslag,'model B radiation validation residue k=4')
plotACFnPACF(rad_modelB_res_k18(validation_start:validation_stop),reslag,'model B radiation validation residue k=18')

var_valid=var(validation_radiation);
fprintf('\n Prediction results of radiation validation for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(validation_start:validation_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(rad_modelB_res_k4(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation validations data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k4(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(validation_start:validation_stop))/var_valid)*100 )
%% B.2.2.1(b) input without k=4
figure;
plot(validation_days,validation_radiation,'k','Linewidth',1.5)
hold on
plot(validation_days,[rad_modelB_k1(validation_start:validation_stop),rad_modelB_k3(validation_start:validation_stop),rad_modelB_k18(validation_start:validation_stop),naive_rad(validation_start:validation_stop)])
title('Model B predictions of radiation validation data')
legend('radiation','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(validation_days,[rad_modelB_res_k1(validation_start:validation_stop),rad_modelB_res_k3(validation_start:validation_stop),rad_modelB_res_k18(validation_start:validation_stop),naive_rad_res(validation_start:validation_stop)])
title('Residues of Model B predictions predictions of validation data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radition error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(validation_start:validation_stop),reslag,'model B radiation validation residue k=1')
plotACFnPACF(rad_modelB_res_k3(validation_start:validation_stop),reslag,'model B radiation validation residue k=3')
plotACFnPACF(rad_modelB_res_k18(validation_start:validation_stop),reslag,'model B radiation validation residue k=18')

var_valid=var(validation_radiation);
fprintf('\n Prediction results of radiation validation for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation validations data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(validation_start:validation_stop))/var_valid)*100 )
%% B.2.2.2 Output
figure;
plot(validation_days,[validation_data,temp_modelB_k1(validation_start:validation_stop),temp_modelB_k3(validation_start:validation_stop),temp_modelB_k4(validation_start:validation_stop),temp_modelB_k18(validation_start:validation_stop),naive(validation_start:validation_stop)])
title('Model B predictions of validation data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(validation_days,[temp_modelB_res_k1(validation_start:validation_stop),temp_modelB_res_k3(validation_start:validation_stop),temp_modelB_res_k4(validation_start:validation_stop),temp_modelB_res_k18(validation_start:validation_stop),naive_res(validation_start:validation_stop)])
title('Residues of Model B predictions of validation data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(validation_start:validation_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(validation_start:validation_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k4(validation_start:validation_stop),reslag,'model B validation residue k=4');
plotACFnPACF(temp_modelB_res_k18(validation_start:validation_stop),reslag,'model B validation residue k=18');

var_valid=var(validation_data);
fprintf('\n Prediction results of validation for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(validation_start:validation_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(temp_modelB_res_k4(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the validations data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k4(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(validation_start:validation_stop))/var_valid)*100 )
%% B.2.2.2 Output without k=4
figure;
plot(validation_days,validation_data,'k','Linewidth',1.5)
hold on
plot(validation_days,[validation_data,temp_modelB_k1(validation_start:validation_stop),temp_modelB_k3(validation_start:validation_stop),temp_modelB_k18(validation_start:validation_stop),naive(validation_start:validation_stop)])
title('Model B predictions of validation data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(validation_days,[temp_modelB_res_k1(validation_start:validation_stop),temp_modelB_res_k3(validation_start:validation_stop),temp_modelB_res_k18(validation_start:validation_stop),naive_res(validation_start:validation_stop)])
title('Residues of Model B predictions of validation data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(validation_start:validation_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(validation_start:validation_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k18(validation_start:validation_stop),reslag,'model B validation residue k=18');

var_valid=var(validation_data);
fprintf('\n Prediction results of validation for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(validation_start:validation_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(temp_modelB_res_k4(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the validations data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(validation_start:validation_stop))/var_valid)*100 )
%% B.2.3 forming prediction for test periods
%% B.2.3.1 Test 1 
%% B.2.3.1.1 input
figure;
plot(test1_days,[test_radiation1,rad_modelB_k1(test1_start:test1_stop),rad_modelB_k3(test1_start:test1_stop),rad_modelB_k4(test1_start:test1_stop),rad_modelB_k18(test1_start:test1_stop),naive_rad(test1_start:test1_stop)])
title('Model B predictions of radiation test1 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test1_days,[rad_modelB_res_k1(test1_start:test1_stop),rad_modelB_res_k3(test1_start:test1_stop),rad_modelB_res_k4(test1_start:test1_stop),rad_modelB_res_k18(test1_start:test1_stop),naive_rad_res(test1_start:test1_stop)])
title('Residues of Model B predictions of radiation test1 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test1_start:test1_stop),reslag,'model B radiation test1 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test1_start:test1_stop),reslag,'model B radiation test1 residue k=3');
plotACFnPACF(rad_modelB_res_k4(test1_start:test1_stop),reslag,'model B radiation test1 residue k=4');
plotACFnPACF(rad_modelB_res_k18(test1_start:test1_stop),reslag,'model B radiation test1 residue k=18');

var_test1_rad=var(test_radiation1);
fprintf('\n Prediction results of radiation test1 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test1_start:test1_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(rad_modelB_res_k4(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test1 data is    %7.2f\n', var_test1_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop))/var_test1_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k4(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test1_start:test1_stop))/var_test1_rad)*100 )
%% B.2.3.1.1(b) input without k=4
figure;
plot(test1_days,test_radiation1,'k','Linewidth',1.5)
hold on
plot(test1_days,[rad_modelB_k1(test1_start:test1_stop),rad_modelB_k3(test1_start:test1_stop),rad_modelB_k18(test1_start:test1_stop),naive_rad(test1_start:test1_stop)])
title('Model B predictions of radiation test1 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test1_days,[rad_modelB_res_k1(test1_start:test1_stop),rad_modelB_res_k3(test1_start:test1_stop),rad_modelB_res_k18(test1_start:test1_stop),naive_rad_res(test1_start:test1_stop)])
title('Residues of Model B predictions of radiation test1 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test1_start:test1_stop),reslag,'model B radiation test1 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test1_start:test1_stop),reslag,'model B radiation test1 residue k=3');
plotACFnPACF(rad_modelB_res_k18(test1_start:test1_stop),reslag,'model B radiation test1 residue k=18');

var_test1_rad=var(test_radiation1);
fprintf('\n Prediction results of radiation test1 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test1 data is    %7.2f\n', var_test1_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop))/var_test1_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test1_start:test1_stop))/var_test1_rad)*100 )
%% B.2.2.2(b) Output without k=4
figure;
plot(test1_days,test1_data,'k','Linewidth',1.5)
hold on
plot(test1_days,[temp_modelB_k1(test1_start:test1_stop),temp_modelB_k3(test1_start:test1_stop),temp_modelB_k18(test1_start:test1_stop),naive(test1_start:test1_stop)])
title('Model B predictions of test1 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test1_days,[temp_modelB_res_k1(test1_start:test1_stop),temp_modelB_res_k3(test1_start:test1_stop),temp_modelB_res_k18(test1_start:test1_stop),naive_res(test1_start:test1_stop)])
title('Residues of Model B predictions of test1 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test1_start:test1_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test1_start:test1_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k18(test1_start:test1_stop),reslag,'model B validation residue k=18');

var_test1=var(test1_data);
fprintf('\n Prediction results of test1 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the test1 data is    %7.2f\n', var_test1 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop))/var_test1 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test1_start:test1_stop))/var_test1)*100 )
%% B.2.2.2
figure;
plot(test1_days,[test1_data,temp_modelB_k1(test1_start:test1_stop),temp_modelB_k3(test1_start:test1_stop),temp_modelB_k4(test1_start:test1_stop),temp_modelB_k18(test1_start:test1_stop),naive(test1_start:test1_stop)])
title('Model B predictions of test1 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test1_days,[temp_modelB_res_k1(test1_start:test1_stop),temp_modelB_res_k3(test1_start:test1_stop),temp_modelB_res_k4(test1_start:test1_stop),temp_modelB_res_k18(test1_start:test1_stop),naive_res(test1_start:test1_stop)])
title('Residues of Model B predictions of test1 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test1_start:test1_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test1_start:test1_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k4(test1_start:test1_stop),reslag,'model B validation residue k=4');
plotACFnPACF(temp_modelB_res_k18(test1_start:test1_stop),reslag,'model B validation residue k=18');

var_test1=var(test1_data);
fprintf('\n Prediction results of test1 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test1_start:test1_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(temp_modelB_res_k4(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the test1 data is    %7.2f\n', var_test1 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test1_start:test1_stop))/var_test1 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop))/var_test1 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k4(test1_start:test1_stop))/var_test1)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test1_start:test1_stop))/var_test1)*100 )
%% B.2.4.1 Test 2 
%% B.2.4.1.1 input
figure;
plot(test2_days,[test_radiation2,rad_modelB_k1(test2_start:test2_stop),rad_modelB_k3(test2_start:test2_stop),rad_modelB_k4(test2_start:test2_stop),rad_modelB_k18(test2_start:test2_stop),naive_rad(test2_start:test2_stop)])
title('Model B predictions of radiation test2 data')
legend('radiation','k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test2_days,[rad_modelB_res_k1(test2_start:test2_stop),rad_modelB_res_k3(test2_start:test2_stop),rad_modelB_res_k4(test2_start:test2_stop),rad_modelB_res_k18(test2_start:test2_stop),naive_rad_res(test2_start:test2_stop)])
title('Residues of Model B predictions of radiation test2 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test2_start:test2_stop),reslag,'model B radiation test2 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test2_start:test2_stop),reslag,'model B radiation test2 residue k=3');
plotACFnPACF(rad_modelB_res_k4(test2_start:test2_stop),reslag,'model B radiation test2 residue k=4');
plotACFnPACF(rad_modelB_res_k18(test2_start:test2_stop),reslag,'model B radiation test2 residue k=18');

var_test2_rad=var(test_radiation2);
fprintf('\n Prediction results of radiation test2 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(rad_modelB_res_k4(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test2 data is    %7.2f\n', var_test2_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop))/var_test2_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k4(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test2_start:test2_stop))/var_test2_rad)*100 )
%% B.2.4.1.1(b) input without k=4
figure;
plot(test2_days,test_radiation2,'k','Linewidth',1.5)
hold on
plot(test2_days,[rad_modelB_k1(test2_start:test2_stop),rad_modelB_k3(test2_start:test2_stop),rad_modelB_k18(test2_start:test2_stop),naive_rad(test2_start:test2_stop)])
title('Model B predictions of radiation test2 data')
legend('radiation','k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test2_days,[rad_modelB_res_k1(test2_start:test2_stop),rad_modelB_res_k3(test2_start:test2_stop),rad_modelB_res_k18(test2_start:test2_stop),naive_rad_res(test2_start:test2_stop)])
title('Residues of Model B predictions of radiation test2 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test2_start:test2_stop),reslag,'model B radiation test2 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test2_start:test2_stop),reslag,'model B radiation test2 residue k=3');
plotACFnPACF(rad_modelB_res_k18(test2_start:test2_stop),reslag,'model B radiation test2 residue k=18');

var_test2_rad=var(test_radiation2);
fprintf('\n Prediction results of radiation test2 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test2 data is    %7.2f\n', var_test2_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop))/var_test2_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test2_start:test2_stop))/var_test2_rad)*100 )
%% B.2.4.1.2 Output
figure;
plot(test2_days,[test2_data,temp_modelB_k1(test2_start:test2_stop),temp_modelB_k3(test2_start:test2_stop),temp_modelB_k4(test2_start:test2_stop),temp_modelB_k18(test2_start:test2_stop),naive(test2_start:test2_stop)])
title('Model B predictions of test2 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test2_days,[temp_modelB_res_k1(test2_start:test2_stop),temp_modelB_res_k3(test2_start:test2_stop),temp_modelB_res_k4(test2_start:test2_stop),temp_modelB_res_k18(test2_start:test2_stop),naive_res(test2_start:test2_stop)])
title('Residues of Model B predictions of test2 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test2_start:test2_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test2_start:test2_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k4(test2_start:test2_stop),reslag,'model B validation residue k=4');
plotACFnPACF(temp_modelB_res_k18(test2_start:test2_stop),reslag,'model B validation residue k=18');

var_test2=var(test2_data);
fprintf('\n Prediction results of test2 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(temp_modelB_res_k4(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test2 data is    %7.2f\n', var_test2 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop))/var_test2 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k4(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test2_start:test2_stop))/var_test2)*100 )
%% B.2.4.1.2(b) Output without k=4
figure;
plot(test2_days,test2_data,'k','Linewidth',1.5)
hold on
plot(test2_days,[temp_modelB_k1(test2_start:test2_stop),temp_modelB_k3(test2_start:test2_stop),temp_modelB_k18(test2_start:test2_stop),naive(test2_start:test2_stop)])
title('Model B predictions of test2 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test2_days,[temp_modelB_res_k1(test2_start:test2_stop),temp_modelB_res_k3(test2_start:test2_stop),temp_modelB_res_k18(test2_start:test2_stop),naive_res(test2_start:test2_stop)])
title('Residues of Model B predictions of test2 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test2_start:test2_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test2_start:test2_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k18(test2_start:test2_stop),reslag,'model B validation residue k=18');

var_test2=var(test2_data);
fprintf('\n Prediction results of test2 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test2 data is    %7.2f\n', var_test2 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test2_start:test2_stop))/var_test2 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop))/var_test2 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test2_start:test2_stop))/var_test2)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test2_start:test2_stop))/var_test2)*100 )
%% B.2.5.1 Test 3 
%% B.2.5.1.1b(b) input without k=4
figure;
plot(test3_days,test_radiation3,'k','Linewidth',1.5)
hold on
plot(test3_days,[rad_modelB_k1(test3_start:test3_stop),rad_modelB_k3(test3_start:test3_stop),rad_modelB_k18(test3_start:test3_stop),naive_rad(test3_start:test3_stop)])
title('Model B predictions of radiation test3 data')
legend('radiation','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test3_days,[rad_modelB_res_k1(test3_start:test3_stop),rad_modelB_res_k3(test3_start:test3_stop),rad_modelB_res_k18(test3_start:test3_stop),naive_rad_res(test3_start:test3_stop)])
title('Residues of Model B predictions of radiation test3 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Raditiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test3_start:test3_stop),reslag,'model B radiation test3 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test3_start:test3_stop),reslag,'model B radiation test3 residue k=3');
plotACFnPACF(rad_modelB_res_k18(test3_start:test3_stop),reslag,'model B radiation test3 residue k=18');

var_test3_rad=var(test_radiation3);
fprintf('\n Prediction results of radiation test3 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test3_start:test3_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test3_start:test3_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test3_start:test3_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test3 data is    %7.2f\n', var_test3_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test3 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop))/var_test3_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test3_start:test3_stop))/var_test3_rad)*100) 
%% B.2.5.1.1 input
figure;
plot(test3_days,[test_radiation3,rad_modelB_k1(test3_start:test3_stop),rad_modelB_k3(test3_start:test3_stop),rad_modelB_k4(test3_start:test3_stop),rad_modelB_k18(test3_start:test3_stop),naive_rad(test3_start:test3_stop)])
title('Model B predictions of radiation test3 data')
legend('radiation','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test3_days,[rad_modelB_res_k1(test3_start:test3_stop),rad_modelB_res_k3(test3_start:test3_stop),rad_modelB_res_k4(test3_start:test3_stop),rad_modelB_res_k18(test3_start:test3_stop),naive_rad_res(test3_start:test3_stop)])
title('Residues of Model B predictions of radiation test3 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Raditiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelB_res_k1(test3_start:test3_stop),reslag,'model B radiation test3 residue k=1');
plotACFnPACF(rad_modelB_res_k3(test3_start:test3_stop),reslag,'model B radiation test3 residue k=3');
plotACFnPACF(rad_modelB_res_k4(test3_start:test3_stop),reslag,'model B radiation test3 residue k=4');
plotACFnPACF(rad_modelB_res_k18(test3_start:test3_stop),reslag,'model B radiation test3 residue k=18');

var_test3_rad=var(test_radiation3);
fprintf('\n Prediction results of radiation test3 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelB_res_k1(test3_start:test3_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelB_res_k3(test3_start:test3_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(rad_modelB_res_k4(test3_start:test3_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelB_res_k18(test3_start:test3_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test3 data is    %7.2f\n', var_test3_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test3 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelB_res_k1(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelB_res_k3(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(rad_modelB_res_k4(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelB_res_k18(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop))/var_test3_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k1(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k3(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k4(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelB_res_k18(test3_start:test3_stop))/var_test3_rad)*100 )
%% B.2.2.2 Output
figure;
plot(test3_days,[test3_data,temp_modelB_k1(test3_start:test3_stop),temp_modelB_k3(test3_start:test3_stop),temp_modelB_k4(test3_start:test3_stop),temp_modelB_k18(test3_start:test3_stop),naive(test3_start:test3_stop)])
title('Model B predictions of test3 data')
legend('temperature','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test3_days,[temp_modelB_res_k1(test3_start:test3_stop),temp_modelB_res_k3(test3_start:test3_stop),temp_modelB_res_k4(test3_start:test3_stop),temp_modelB_res_k18(test3_start:test3_stop),naive_res(test3_start:test3_stop)])
title('Residues of Model B predictions of test3 data')
legend('k=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test3_start:test3_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test3_start:test3_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k4(test3_start:test3_stop),reslag,'model B validation residue k=4');
plotACFnPACF(temp_modelB_res_k18(test3_start:test3_stop),reslag,'model B validation residue k=18');

var_test3=var(test3_data);
fprintf('\n Prediction results of test3 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 4-step: \n')
figure;
whitenessTest(temp_modelB_res_k4(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test3 data is    %7.2f\n', var_test3 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test3 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 4-step prediction residual is %7.2f\n', var(temp_modelB_res_k4(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop))/var_test3 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 4-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k4(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test3_start:test3_stop))/var_test3)*100 )

%% B.2.2.2(b) Output without k=4
figure;
plot(test3_days,test3_data,'k','Linewidth',1.5)
hold on
plot(test3_days,[temp_modelB_k1(test3_start:test3_stop),temp_modelB_k3(test3_start:test3_stop),temp_modelB_k18(test3_start:test3_stop),naive(test3_start:test3_stop)])
title('Model B predictions of test3 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature(°C)');
figure;
plot(test3_days,[temp_modelB_res_k1(test3_start:test3_stop),temp_modelB_res_k3(test3_start:test3_stop),temp_modelB_res_k18(test3_start:test3_stop),naive_res(test3_start:test3_stop)])
title('Residues of Model B predictions of test3 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Temperature error(°C)');
hold off
plotACFnPACF(temp_modelB_res_k1(test3_start:test3_stop),reslag,'model B validation residue k=1');
plotACFnPACF(temp_modelB_res_k3(test3_start:test3_stop),reslag,'model B validation residue k=3');
plotACFnPACF(temp_modelB_res_k18(test3_start:test3_stop),reslag,'model B validation residue k=18');

var_test3=var(test3_data);
fprintf('\n Prediction results of test3 for model B:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(temp_modelB_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(temp_modelB_res_k3(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(temp_modelB_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the test3 data is    %7.2f\n', var_test3 )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test3 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(temp_modelB_res_k1(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(temp_modelB_res_k3(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(temp_modelB_res_k18(test3_start:test3_stop))/var_test3 );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop))/var_test3 );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k1(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k3(test3_start:test3_stop))/var_test3)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(temp_modelB_res_k18(test3_start:test3_stop))/var_test3)*100 )
%% Part C, Section 1:  Using Kalman Filter to Estimate BJ Model from Part B 

%% C.1.1  Initialize data and parameters
hold_radiation=radiation/100;%to avoid redefinition and to ensure KB is not 0
y = temperature; 
u = hold_radiation;  
N = length(y);        

% Initial model coefficients from the BJ model
KA = [1, -1.59, 0.5996]; % D(z) = 1 - 1.59z^-1 + 0.5996z^-2
KB = [0.16];         % B(z) = 0.007012
% Define number of parameters
noPar = length(KA) - 1 + length(KB); % [-KA(2:end), KB(1:end)]
%noPar = 2; % to test without KB ; result is that KB has little bering on
%prediction
% Initializing Kalman filter variables
xt = zeros(noPar, N);      
xt(:, 1) = [KA(2:end), KB];
%xt(:,1) = [ KA(2), KA(3) ];%to test without KB
Rx_t1 = eye(noPar) * 0.9;   %% Adjust up if Poor initial estimates, adjust down if Good prior knowledge
A = eye(noPar);            
Rw = 1;                     %% Adjust up if Residuals too large, adjust dowm if Filter reacts too fast to noise
Re = 1e-7 * eye(noPar);     %% Adjust up if Model deviates from data, adjust down if Parameters fluctuate too much
h_et = zeros(N, 1);        
yhatK = zeros(N, 1);       
xStd = zeros(noPar, N);    

%% C.1.2  Build Kalman Loop for Prediction
startInd = max(length(KA), length(KB)) + 1;
% Kalman Filter Loop
for t = startInd:N
    x_t1 = A * xt(:, t - 1); % x_{t|t-1} = A x_{t-1|t-1}
    C = [-y(t - 1), -y(t - 2), u(t)]; % [ -KA(2) -KA(3) KB(1) ]
    %C = [-y(t - 1), -y(t - 2)];%[ -KA(2) -KA(3) ]
    yhatK(t) = C * x_t1;
    Ry = C * Rx_t1 * C' + Rw;          
    Kt = Rx_t1 * C' / Ry;             
    h_et(t) = y(t) - yhatK(t);        
    xt(:, t) = x_t1 + Kt * h_et(t);   
    Rx_t = Rx_t1 - Kt * Ry * Kt';     
    Rx_t1 = A * Rx_t * A' + Re;       
    xStd(:, t) = sqrt(diag(Rx_t));
end
% OBS Do not run twice by mistake
%%  C.1.3 Analyze and compare results
% Compare estimated parameters over time
figure;
plotWithConf(1:N, xt', xStd', []);
line([startInd startInd], [-2 2], 'Color', 'red', 'LineStyle', ':');
title('Kalman Filter: Estimated Parameters');
xlabel('Time');
ylabel('Parameter Value');
%xticks()
%xticklabels(date_time,'manual')
grid on;



% Display final estimates
disp('Final Parameter Estimates:');
for i = 1:noPar
    fprintf('Estimated Parameter %d: %5.4f (+/- %5.4f)\n', i, xt(i, end), xStd(i, end));
end

% Plot the prediction compared to the real data
figure;
plot(y, 'b', 'LineWidth', 1.5); % Real data (output)
hold on;
plot(yhatK, 'r--', 'LineWidth', 1.5); % Predicted data
title('Real Data vs. Prediction (Kalman Filter)');
xlabel('Time');
ylabel('Temperature(°C)');
legend('Real Data', 'Prediction', 'Location', 'Best');
grid on;
hold off;


% Residual analysis
eK = y - yhatK; % Residuals
figure;
plot(eK);
title('Prediction Residuals');
xlabel('Time');
ylabel('Temperature error(°C)');
grid on;

figure;
acf(eK(startInd:end), 30, 0.05, 1);
title('ACF of Residuals');
checkIfWhite(eK(startInd:end));
fprintf('\n Residue variance is %4.2f \n ',var(eK))
% save prediction and residual for later use

yhatK1=yhatK;
eK1=eK;
%% C.1.4 Doing a 4-step as well 
startInd = max(length(KA), length(KB)) + 4;% think this is only needed change

% Kalman Filter Loop
for t = startInd:N
    x_t1 = A * xt(:, t - 4); % x_{t|t-1} = A x_{t-1|t-1} %Think this is only needed change
    C = [-y(t - 1), -y(t - 2), u(t)]; % [ -KA(2) -KA(3) KB(1) ]
    yhatK(t) = C * x_t1;
    Ry = C * Rx_t1 * C' + Rw;          
    Kt = Rx_t1 * C' / Ry;             
    h_et(t) = y(t) - yhatK(t);        
    xt(:, t) = x_t1 + Kt * h_et(t);   
    Rx_t = Rx_t1 - Kt * Ry * Kt';     
    Rx_t1 = A * Rx_t * A' + Re;       
    xStd(:, t) = sqrt(diag(Rx_t));
end

%%  C.1.5 Analyze and compare results of 4-step
% Compare estimated parameters over time
figure;
plotWithConf(1:N, xt', xStd', []);
line([startInd startInd], [-2 2], 'Color', 'red', 'LineStyle', ':');
title('Kalman Filter: Estimated Parameters');
xlabel('Time');
ylabel('Parameter Value');
grid on;

% Display final estimates
disp('Final Parameter Estimates:');
for i = 1:noPar
    fprintf('Estimated Parameter %d: %5.4f (+/- %5.4f)\n', i, xt(i, end), xStd(i, end));
end

% Plot the prediction compared to the real data
figure;
plot(y, 'b', 'LineWidth', 1.5); % Real data (output)
hold on;
plot(yhatK, 'r--', 'LineWidth', 1.5); % Predicted data
title('Real Data vs. Prediction (Kalman Filter)');
xlabel('Time');
ylabel('Temperature(°C)');
legend('Real Data', 'Prediction', 'Location', 'Best');
grid on;
hold off;


% Residual analysis
eK = y - yhatK; % Residuals
figure;
plot(eK);
title('Prediction Residuals');
xlabel('Time');
ylabel('Temperature error(°C)');
grid on;

figure;
acf(eK(startInd:end), 30, 0.05, 1);
title('ACF of Residuals');
checkIfWhite(eK(startInd:end));
 
% save prediction and residual for later use
yhatK4=yhatK;
eK4=eK;
%% C.1.6 3- and 18-step predictions

%% C.1.6.1 3-step prediction
% Kalman Filter Loop
startInd = max(length(KA), length(KB)) + 3;% think this is only needed change
for t = startInd:N
    x_t1 = A * xt(:, t - 3); % x_{t|t-1} = A x_{t-1|t-1} %Think this is only needed change
    C = [-y(t - 1), -y(t - 2), u(t)]; % [ -KA(2) -KA(3) KB(1) ]
    yhatK(t) = C * x_t1;
    Ry = C * Rx_t1 * C' + Rw;          
    Kt = Rx_t1 * C' / Ry;             
    h_et(t) = y(t) - yhatK(t);        
    xt(:, t) = x_t1 + Kt * h_et(t);   
    Rx_t = Rx_t1 - Kt * Ry * Kt';     
    Rx_t1 = A * Rx_t * A' + Re;       
    xStd(:, t) = sqrt(diag(Rx_t));
end
eK = y - yhatK; % Residuals

yhatK3=yhatK;
eK3=eK;
% Display final estimates
disp('Final Parameter Estimates:');
for i = 1:noPar
    fprintf('Estimated Parameter %d: %5.4f (+/- %5.4f)\n', i, xt(i, end), xStd(i, end));
end
%%
% Compare estimated parameters over time
figure;
plotWithConf(1:N, xt', xStd', []);
line([startInd startInd], [-2 2], 'Color', 'red', 'LineStyle', ':');
title('Kalman Filter: Estimated Parameters');
xlabel('Time');
ylabel('Parameter Value');
grid on;



%% C.1.6.1 18-step prediction
startInd = max(length(KA), length(KB)) + 18;% think this is only needed change
% Kalman Filter Loop
for t = startInd:N
    x_t1 = A * xt(:, t - 18); % x_{t|t-1} = A x_{t-1|t-1} %Think this is only needed change
    C = [-y(t - 1), -y(t - 2), u(t)]; % [ -KA(2) -KA(3) KB(1) ]
    yhatK(t) = C * x_t1;
    Ry = C * Rx_t1 * C' + Rw;          
    Kt = Rx_t1 * C' / Ry;             
    h_et(t) = y(t) - yhatK(t);        
    xt(:, t) = x_t1 + Kt * h_et(t);   
    Rx_t = Rx_t1 - Kt * Ry * Kt';     
    Rx_t1 = A * Rx_t * A' + Re;       
    xStd(:, t) = sqrt(diag(Rx_t));
end
eK = y - yhatK; % Residuals

yhatK18=yhatK;
eK18=eK;

% Display final estimates
disp('Final Parameter Estimates:');
for i = 1:noPar
    fprintf('Estimated Parameter %d: %5.4f (+/- %5.4f)\n', i, xt(i, end), xStd(i, end));
end

%%

% Compare estimated parameters over time
figure;
plotWithConf(1:N, xt', xStd', []);
line([startInd startInd], [-2 2], 'Color', 'red', 'LineStyle', ':');
title('Kalman Filter: Estimated Parameters');
xlabel('Time');
ylabel('Parameter Value');
grid on;
%% C.2.1 Extracting prediction for valdiation and tests
%% C.2.2 Validation without k=4
val_predK1=yhatK1(validation_start:validation_stop);
val_predK18=yhatK3(validation_start:validation_stop);
%val_predK4=yhatK4(validation_start:validation_stop);
val_predK3=yhatK18(validation_start:validation_stop);
val_eK1= eK1(validation_start:validation_stop);
val_eK3= eK3(validation_start:validation_stop);
%val_eK4= eK4(validation_start:validation_stop);
val_eK18= eK18(validation_start:validation_stop);
var_valid=var(validation_data);
figure;
plot(validation_days,validation_data,'k','LineWidth',1.5)
hold on
plot(validation_days,[ val_predK1,val_predK3,val_predK18])
title('Model C: Kalman prediction of validation data')
xlabel('Time');
ylabel('Temperature(°C)');
legend('temperature', '1-step prediction','3-step prediction','18-step prediction')
hold off

figure;
plot(validation_days,[val_eK1,val_eK3,val_eK18])
title('model C: Residue of Kalman prediction of validation')
legend( '1-step prediction','3-step prediction','18-step prediction')
xlabel('Time');
ylabel('Temperature error(°C)');
fprintf('\n Prediction results of validation for model C:\n')

fprintf('Prediction results for output of validation:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(val_eK1)
fprintf('\n For 3-step: \n')
figure;
whitenessTest(val_eK3)
fprintf('\n For 18-step: \n')
figure;
whitenessTest(val_eK18)
fprintf('variance:\n')
fprintf('  The variance of the validations data is    %7.2f\n', var(validation_data)) 
fprintf('  The variance of the prediction k=1 residual is %7.2f\n', var(val_eK1) )
fprintf('  The variance of the prediction k=3 residual is %7.2f\n', var(val_eK3))
%fprintf('  The variance of the prediction k=4 residual is %7.2f\n', var(val_eK4))
fprintf('  The variance of the prediction k=18 residual is %7.2f\n', var(val_eK18))
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the validations data is    %7.2f\n', 1.00) 
fprintf('  The normalized variance of the prediction k=1 residual is %7.2f\n', var(val_eK1)/var_valid )
fprintf('  The normalized variance of the prediction k=3 residual is %7.2f\n', var(val_eK3)/var_valid)
%fprintf('  The normalized variance of the prediction k=4 residual is %7.2f\n', var(val_eK4)/var_valid)
fprintf('  The normalized variance of the prediction k=18 residual is %7.2f\n', var(val_eK18)/var_valid)
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(validation_start:validation_stop))/var_valid )
fprintf('How much is explained:\n')
fprintf('  The prediction k=1 explains %3.1f%% of the variance\n',  (1-var(val_eK1)/var(validation_data))*100 )
fprintf('  The prediction k=3 explains %3.1f%% of the variance\n',  (1-var(val_eK3)/var(validation_data))*100 )
%fprintf('  The prediction k=4 explains %3.1f%% of the variance\n',  (1-var(val_eK4)/var(validation_data))*100 )
fprintf('  The prediction k=18 explains %3.1f%% of the variance\n',  (1-var(val_eK18)/var(validation_data))*100 )
%% C.2.3 tests
%% C.2.3.1 test1 without k=4
test1_predK1=yhatK1(test1_start:test1_stop);
test1_predK3=yhatK3(test1_start:test1_stop);
%test1_predK4=yhatK4(test1_start:test1_stop);
test1_predK18=yhatK18(test1_start:test1_stop);
test1_eK1= eK1(test1_start:test1_stop);
test1_eK3= eK3(test1_start:test1_stop);
%test1_eK4= eK4(test1_start:test1_stop);
test1_eK18= eK18(test1_start:test1_stop);
figure;
plot(test1_days,test1_data,'k','LineWidth',1.5)
hold on
plot(test1_days,[ test1_predK1,test1_predK3,test1_predK18])
title('Kalman prediction of test1')
xlabel('Time');
ylabel('Temperature(°C)');
legend('temperature', '1-step prediction','3-step prediction','18-step prediction')

figure;
plot(test1_days,[test1_eK1,test1_eK3,test1_eK18])
title('Residue of Kalman prediction of test1')
xlabel('Time');
ylabel('Temperature error(°C)');
legend( '1-step prediction','3-step prediction','18-step prediction')
var_test1=var(test1_data);
fprintf('Prediction results for output of test1:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(test1_eK1)
fprintf('\n For 3-step: \n')
figure;
whitenessTest(test1_eK3)
%fprintf('\n For 4-step: \n')
%figure;
%whitenessTest(test1_eK4)
fprintf('\n For 18-step: \n')
figure;
whitenessTest(test1_eK18)
var_test1=var(test1_data);
fprintf('  The variance of the test1 data is    %7.2f\n', var(test1_data)) 
fprintf('  The variance of the prediction k=1 residual is %7.2f\n', var(test1_eK1) )
fprintf('  The variance of the prediction k=3 residual is %7.2f\n', var(test1_eK3))
%fprintf('  The variance of the prediction k=4 residual is %7.2f\n', var(test1_eK4))
fprintf('  The variance of the prediction k=18 residual is %7.2f\n', var(test1_eK18))
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test1 data is    %7.2f\n', 1.00) 
fprintf('  The normalized variance of the prediction k=1 residual is %7.2f\n', var(test1_eK1)/var_test1 )
fprintf('  The normalized variance of the prediction k=3 residual is %7.2f\n', var(test1_eK3)/var_test1)
%fprintf('  The normalized variance of the prediction k=4 residual is %7.2f\n', var(test1_eK4)/var_test1)
fprintf('  The normalized variance of the prediction k=18 residual is %7.2f\n', var(test1_eK18)/var_test1)
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop))/var_test1 )
fprintf('How much is explained:\n')
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test1_start:test1_stop)) )
fprintf('  The prediction k=1 explains %3.1f%% of the variance\n',  (1-var(test1_eK1)/var_test1)*100 )
fprintf('  The prediction k=3 explains %3.1f%% of the variance\n',  (1-var(test1_eK3)/var_test1)*100 )
%fprintf('  The prediction k=4 explains %3.1f%% of the variance\n',  (1-var(test1_eK4)/var_test1)*100 )
fprintf('  The prediction k=18 explains %3.1f%% of the variance\n',  (1-var(test1_eK18)/var_test1)*100 )
%% C.2.3.2 test2 without k=4
test2_predK1=yhatK1(test2_start:test2_stop);
test2_predK3=yhatK3(test2_start:test2_stop);
%test2_predK4=yhatK4(test2_start:test2_stop);
test2_predK18=yhatK18(test2_start:test2_stop);
test2_eK1= eK1(test2_start:test2_stop);
test2_eK3= eK3(test2_start:test2_stop);
%test2_eK4= eK4(test2_start:test2_stop);
test2_eK18= eK18(test2_start:test2_stop);
figure;
plot(test2_days,test2_data,'k','LineWidth',1.5)
hold on
plot(test2_days,[ test2_predK1,test2_predK3,test2_predK18])
title('Kalman prediction of test2')
xlabel('Time');
ylabel('Temperature(°C)');
legend('temperature', '1-step prediction','3-step prediction''4-step prediction','18-step prediction')

figure;
plot(test2_days,[test2_eK1,test2_eK3,test2_eK18])
title('Residue of Kalman prediction of test2')
xlabel('Time');
ylabel('Temperature error(°C)');
legend( '1-step prediction','3-step prediction','18-step prediction')

fprintf('Prediction results for output of test2:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(test2_eK1)
fprintf('\n For 3-step: \n')
figure;
whitenessTest(test2_eK3)
%fprintf('\n For 4-step: \n')
%figure;
%whitenessTest(test2_eK4)
fprintf('\n For 18-step: \n')
figure;
whitenessTest(test2_eK18)
var_test2=var(test2_data);
fprintf('  The variance of the test2 data is    %7.2f\n', var_test2) 
fprintf('  The variance of the prediction k=1 residual is %7.2f\n', var(test2_eK1) )
fprintf('  The variance of the prediction k=3 residual is %7.2f\n', var(test2_eK3))
%fprintf('  The variance of the prediction k=4 residual is %7.2f\n', var(test2_eK4))
fprintf('  The variance of the prediction k=18 residual is %7.2f\n', var(test2_eK18))
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test2 data is    %7.2f\n', 1.00) 
fprintf('  The normalized variance of the prediction k=1 residual is %7.2f\n', var(test2_eK1)/var_test2 )
fprintf('  The normalized variance of the prediction k=3 residual is %7.2f\n', var(test2_eK3)/var_test2)
%fprintf('  The normalized variance of the prediction k=4 residual is %7.2f\n', var(test2_eK4)/var_test2)
fprintf('  The normalized variance of the prediction k=18 residual is %7.2f\n', var(test2_eK18)/var_test2)
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test2_start:test2_stop))/var_test2 )
fprintf('How much is explained:\n')

fprintf('  The prediction k=1 explains %3.1f%% of the variance\n',  (1-var(test2_eK1)/var_test2)*100 )
fprintf('  The prediction k=3 explains %3.1f%% of the variance\n',  (1-var(test2_eK3)/var_test2)*100 )
%fprintf('  The prediction k=4 explains %3.1f%% of the variance\n',  (1-var(test2_eK4)/var_test2)*100 )
fprintf('  The prediction k=18 explains %3.1f%% of the variance\n',  (1-var(test2_eK18)/var_test2)*100 )

%% C.2.3.3 test3 with out k=4
test3_predK1=yhatK1(test3_start:test3_stop);
test3_predK3=yhatK3(test3_start:test3_stop);
test3_predK4=yhatK4(test3_start:test3_stop);
test3_predK18=yhatK18(test3_start:test3_stop);
test3_eK1= eK1(test3_start:test3_stop);
test3_eK3= eK3(test3_start:test3_stop);
test3_eK4= eK4(test3_start:test3_stop);
test3_eK18= eK18(test3_start:test3_stop);
figure;
plot(test3_days,test3_data,'k','LineWidth',1.5)
hold on
plot(test3_days,[test3_data, test3_predK1,test3_predK3,test3_predK18])
title('Kalman prediction of test3')
xlabel('Time');
ylabel('Temperature(°C)');
legend('temperature', '1-step prediction','3-step prediction','18-step prediction')

figure;
plot(test3_days,[test3_eK1,test3_eK3,test3_eK18])
title('Residue of Kalman prediction of test3')
legend( '1-step prediction','3-step prediction','18-step prediction')
xlabel('Time');
ylabel('Temperature error(°C)');
var_test3=var(test3_data);
fprintf('Prediction results for output of test3:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(test3_eK1)
fprintf('\n For 3-step: \n')
figure;
whitenessTest(test3_eK3)
%fprintf('\n For 4-step: \n')
%figure;
%whitenessTest(test3_eK4)
fprintf('\n For 18-step: \n')
figure;
whitenessTest(test3_eK18)
var_test2=var(test3_data);
fprintf('  The variance of the test3 data is    %7.2f\n', var_test3) 
fprintf('  The variance of the prediction k=1 residual is %7.2f\n', var(test3_eK1) )
fprintf('  The variance of the prediction k=3 residual is %7.2f\n', var(test3_eK3))
%fprintf('  The variance of the prediction k=4 residual is %7.2f\n', var(test3_eK4))
fprintf('  The variance of the prediction k=18 residual is %7.2f\n', var(test3_eK18))
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the test3 data is    %7.2f\n', 1.00) 
fprintf('  The normalized variance of the prediction k=1 residual is %7.2f\n', var(test3_eK1)/var_test3 )
fprintf('  The normalized variance of the prediction k=3 residual is %7.2f\n', var(test3_eK3)/var_test3)
%fprintf('  The normalized variance of the prediction k=4 residual is %7.2f\n', var(test3_eK4)/var_test3)
fprintf('  The normalized variance of the prediction k=18 residual is %7.2f\n', var(test3_eK18)/var_test3)
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_res(test3_start:test3_stop))/var_test3 )
fprintf('How much is explained:\n')

fprintf('  The prediction k=1 explains %3.1f%% of the variance\n',  (1-var(test3_eK1)/var_test3)*100 )
fprintf('  The prediction k=3 explains %3.1f%% of the variance\n',  (1-var(test3_eK3)/var_test3)*100 )
fprintf('  The prediction k=4 explains %3.1f%% of the variance\n',  (1-var(test3_eK4)/var_test3)*100 )
fprintf('  The prediction k=18 explains %3.1f%% of the variance\n',  (1-var(test3_eK18)/var_test3)*100 )
%% Kalman filter on input (that we know is useless but we still need it)
%% Kalman filter on input (that we know it's useless but we still need it for report)
% we shall use code 24 and 22
load('projectData24.mat'); 
radiation = data(:, 6); % Net Radiation (input)
temperature = data(:, 8); % Temperature (output)
year = data(:, 1);         
month = data(:, 2);       
day = data(:, 3);         
hour = data(:, 4);
date_time = datetime(year, month, day, hour, 0, 0);
start_index=34409;
end_index=35752;
%Validation data is choosen as theweek directly after model data
validation_data = temperature(end_index+1:end_index + 24*7 );
validation_days = date_time(end_index+1:end_index+ 24*7 );
validation_start=end_index+1;
validation_stop=end_index+ 24*7;
%test data is chosen as the 4 weeks directly after validation data
Test_length= 24*7*4;
test1_data= temperature(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
test1_days = date_time(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
test1_start=end_index+ 24*7+1;
test1_stop=end_index+ 24*7+Test_length;
%Additional Sections of test data will/can be added such as same period
%next year or summer and spring.
%Test data of same period the year after
test2_data= temperature(start_index+ 365*24 :start_index+ 365*24+Test_length);
test2_days = date_time(start_index+ 365*24 :start_index+ 365*24+Test_length);
test2_start=start_index+ 365*24 ;
test2_stop = start_index+ 365*24+Test_length;
%Test data of following summer, it is expected that model will be worst for
%summer since model is based on a period in the winter.
test3_data= temperature(start_index+ 140*24 :start_index+ 140*24+Test_length);
test3_days = date_time(start_index+ 140*24 :start_index+ 140*24+Test_length);
test3_start =start_index+ 140*24;
test3_stop =start_index+ 140*24+Test_length;
%Validation data is choosen as the week directly after model data
validation_radiation = radiation(end_index+1:end_index + 24*7 );
%test data is chosen as the 4 weeks directly after validation data
Test_length= 24*7*4;
test_radiation1= radiation(end_index+ 24*7+1 :end_index+ 24*7+Test_length);
%Additional Sections of test data will/can be added such as same period
%next year or summer and spring.
%Test data of same period the year after
test_radiation2= radiation(start_index+ 365*24 :start_index+ 365*24+Test_length);
%Test data of following summer, it is expected that model will be worst for
%summer since model is based on a period in the winter.
test_radiation3= radiation(start_index+ 140*24 :start_index+ 140*24+Test_length);

naive_rad=radiation;
naive_rad(1:24)=zeros(24,1);
naive_rad(25:end)=radiation(1:length(temperature)-24);
naive_rad_res=radiation-naive_rad;
reslag=30;
%%
%initial model
A0=[1,-0.99,0.11];
C0=[1 zeros(1,23)-0.80];
seasonal_diff_poly=[1 zeros(1,23) -1];
%y=radiation_seasonal_diff;
%N=length(radiation_seasonal_diff);
% Simulate a process.
rng(1)                                          % Set the seed (just done for the lecture!)
extraN = 100;
N  = 1000;
A0 = conv(seasonal_diff_poly ,[ 1 -0.99,0.11 ]);        % Try changing the polynomials - but note that you need to change the C vector then!
A0(26)=0.74; %estimation graph indicates lower so we take avarage
A0(2)=-0.80; %prameter estimation incates this as higher than -0.99 
C0 = [1 zeros(1,23) -0.87];                            
N= length(radiation);
y=radiation/10; %done to increase stabilty
% Plot realisation.


%% Estimate the unknown parameters using a Kalman filter and form the 3-step prediction.
k  = 3;                                         % k-step prediction.
p0 = 3;                                         % Number of unknowns in the A polynomial (note: this is only the non-zero parameters!).
q0 = 1;                                         % Number of unknowns in the C polynomial (note: this is only the non-zero parameters!).

A     = eye(p0+q0);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
Re    = 1e-8*eye(p0+q0);                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
% 1e-8 in order to increase stability which in this case also 
%decreases tendency to cross considering parameter closeness
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N-k);                       % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N-k,1);                           % Estimated output.
yhatk = zeros(N-k,1);                           % Estimated k-step prediction.
xStd  = zeros(p0+q0,N-k);                       % Stores one std for the one-step prediction.
xStdk = zeros(p0+q0,N-k);                       % Stores one std for the k-step prediction.
for t=27:N-k                                     % We use t-25, so start at t=8. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -y(t-1) -y(t-24) -y(t-25)  h_et(t-24) ];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    Ck = [ -y(t) -y(t-23) -y(t-24) h_et(t-23) ];           % C_{t+1|t}
    yk = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}

    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    Rx_k = Rx_t1;
    for k0=2:k
        Ck = [ -yk -y(t-24+k0) -y(t-25+k0)  h_et(t+k0-23) ]; % C_{t+k|t}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    yhatk(t+k) = yk;                            % Note that this should be stored at t+k.

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );            % This is one std for each of the parameters for the 3-step prediction.
end
%result for 
xStdk3=xStdk;
rad_modelC_k3=yhatk*10;
rad_modelC_res_k3=(y-yhatk)*10;
%% Estimate the unknown parameters using a Kalman filter and form the 18-step prediction.
k  = 18;                                         % k-step prediction.
p0 = 3;                                         % Number of unknowns in the A polynomial (note: this is only the non-zero parameters!).
q0 = 1;                                         % Number of unknowns in the C polynomial (note: this is only the non-zero parameters!).

A     = eye(p0+q0);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
Re    = 1e-8*eye(p0+q0);                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
% 1e-8 in order to increase stability which in this case also 
%decreases tendency to cross considering parameter closeness
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N-k);                       % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N-k,1);                           % Estimated output.
yhatk = zeros(N-k,1);                           % Estimated k-step prediction.
xStd  = zeros(p0+q0,N-k);                       % Stores one std for the one-step prediction.
xStdk = zeros(p0+q0,N-k);                       % Stores one std for the k-step prediction.
for t=27:N-k                                     % We use t-25, so start at t=8. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -y(t-1) -y(t-24) -y(t-25)  h_et(t-24) ];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    Ck = [ -y(t) -y(t-23) -y(t-24) h_et(t-23) ];           % C_{t+1|t}
    yk = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}

    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    Rx_k = Rx_t1;
    for k0=2:k
        Ck = [ -yk -y(t-24+k0) -y(t-25+k0)  h_et(t+k0-23) ]; % C_{t+k|t}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    yhatk(t+k) = yk;                            % Note that this should be stored at t+k.

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );            % This is one std for each of the parameters for the 18-step prediction.
end
%result for 
xStdk18=xStdk;
rad_modelC_k18=yhatk*10;
rad_modelC_res_k18=(y-yhatk)*10;
%% Ensure that rad_hat is of same lenght
rad_ht=zeros(length(y),1);
rad_ht(1:N-k)=yhat;
rad_modelC_res_k1=(rad_ht-y)*10;
rad_modelC_k1= rad_ht*10;
%% Examine the estimated parameters.

trueParams = [A0(2) A0(25) A0(26) C0(25)];   % these are the initial parameters.
figure
plotWithConf( date_time(1:N-k), xt', xStdk', trueParams );
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
ylim([-1.5 1.5])
fprintf('The final values of the estimated parameters are:\n')
for k0=1:length(trueParams)
    fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
end 


%% Validation and test data examination
%% B.2.2.1(b) input without k=4
figure;
plot(validation_days,validation_radiation,'k','Linewidth',1.5)
hold on
plot(validation_days,[rad_modelC_k1(validation_start:validation_stop),rad_modelC_k3(validation_start:validation_stop),rad_modelC_k18(validation_start:validation_stop),naive_rad(validation_start:validation_stop)])
title('Model C predictions of radiation validation data')
legend('radiation','K=1','k=3','k=4','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(validation_days,[rad_modelC_res_k1(validation_start:validation_stop),rad_modelC_res_k3(validation_start:validation_stop),rad_modelC_res_k18(validation_start:validation_stop),naive_rad_res(validation_start:validation_stop)])
title('Residues of Model C predictions predictions of validation data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radition error(W/m^2)');
hold off
plotACFnPACF(rad_modelC_res_k1(validation_start:validation_stop),reslag,'model C radiation validation residue k=1')
plotACFnPACF(rad_modelC_res_k3(validation_start:validation_stop),reslag,'model C radiation validation residue k=3')
plotACFnPACF(rad_modelC_res_k18(validation_start:validation_stop),reslag,'model C radiation validation residue k=18')

var_valid=var(validation_radiation);
fprintf('\n Prediction results of radiation validation for model C:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelC_res_k1(validation_start:validation_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelC_res_k3(validation_start:validation_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelC_res_k18(validation_start:validation_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation validations data is    %7.2f\n', var_valid )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(validation_start:validation_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(validation_start:validation_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(validation_start:validation_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation validations data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(validation_start:validation_stop))/var_valid );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(validation_start:validation_stop))/var_valid );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k1(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k3(validation_start:validation_stop))/var_valid)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k18(validation_start:validation_stop))/var_valid)*100 )

%% B.2.3 forming prediction for test periods
%% B.2.3.1 Test 1 
%% B.2.3.1.1(b) input without k=4
figure;
plot(test1_days,test_radiation1,'k','Linewidth',1.5)
hold on
plot(test1_days,[rad_modelC_k1(test1_start:test1_stop),rad_modelC_k3(test1_start:test1_stop),rad_modelC_k18(test1_start:test1_stop),naive_rad(test1_start:test1_stop)])
title('model C predictions of radiation test1 data')
legend('temperature','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test1_days,[rad_modelC_res_k1(test1_start:test1_stop),rad_modelC_res_k3(test1_start:test1_stop),rad_modelC_res_k18(test1_start:test1_stop),naive_rad_res(test1_start:test1_stop)])
title('Residues of model C predictions of radiation test1 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
hold off
plotACFnPACF(rad_modelC_res_k1(test1_start:test1_stop),reslag,'model C radiation test1 residue k=1');
plotACFnPACF(rad_modelC_res_k3(test1_start:test1_stop),reslag,'model C radiation test1 residue k=3');
plotACFnPACF(rad_modelC_res_k18(test1_start:test1_stop),reslag,'model C radiation test1 residue k=18');

var_test1_rad=var(test_radiation1);
fprintf('\n Prediction results of radiation test1 for model C:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelC_res_k1(test1_start:test1_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelC_res_k3(test1_start:test1_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelC_res_k18(test1_start:test1_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test1 data is    %7.2f\n', var_test1_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test1_start:test1_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test1_start:test1_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test1_start:test1_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test1 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test1_start:test1_stop))/var_test1_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test1_start:test1_stop))/var_test1_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k1(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k3(test1_start:test1_stop))/var_test1_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k18(test1_start:test1_stop))/var_test1_rad)*100 )
%% B.2.4.1 Test 2 
%% B.2.4.1.1(b) input without k=4
figure;
plot(test2_days,test_radiation2,'k','Linewidth',1.5)
hold on
plot(test2_days,[rad_modelC_k1(test2_start:test2_stop),rad_modelC_k3(test2_start:test2_stop),rad_modelC_k18(test2_start:test2_stop),naive_rad(test2_start:test2_stop)])
title('model C predictions of radiation test2 data')
legend('radiation','k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test2_days,[rad_modelC_res_k1(test2_start:test2_stop),rad_modelC_res_k3(test2_start:test2_stop),rad_modelC_res_k18(test2_start:test2_stop),naive_rad_res(test2_start:test2_stop)])
title('Residues of model C predictions of radiation test2 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelC_res_k1(test2_start:test2_stop),reslag,'model C radiation test2 residue k=1');
plotACFnPACF(rad_modelC_res_k3(test2_start:test2_stop),reslag,'model C radiation test2 residue k=3');
plotACFnPACF(rad_modelC_res_k18(test2_start:test2_stop),reslag,'model C radiation test2 residue k=18');

var_test2_rad=var(test_radiation2);
fprintf('\n Prediction results of radiation test2 for model C:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelC_res_k1(test2_start:test2_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelC_res_k3(test2_start:test2_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelC_res_k18(test2_start:test2_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test2 data is    %7.2f\n', var_test2_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test2_start:test2_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test2_start:test2_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test2_start:test2_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test2 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test2_start:test2_stop))/var_test2_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test2_start:test2_stop))/var_test2_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k1(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k3(test2_start:test2_stop))/var_test2_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k18(test2_start:test2_stop))/var_test2_rad)*100 )

%% C.2.5.1.1b(b) input without k=4
figure;
plot(test3_days,test_radiation3,'k','Linewidth',1.5)
hold on
plot(test3_days,[rad_modelC_k1(test3_start:test3_stop),rad_modelC_k3(test3_start:test3_stop),rad_modelC_k18(test3_start:test3_stop),naive_rad(test3_start:test3_stop)])
title('model C predictions of radiation test3 data')
legend('radiation','K=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Radiation(W/m^2)');
figure;
plot(test3_days,[rad_modelC_res_k1(test3_start:test3_stop),rad_modelC_res_k3(test3_start:test3_stop),rad_modelC_res_k18(test3_start:test3_stop),naive_rad_res(test3_start:test3_stop)])
title('Residues of model C predictions of radiation test3 data')
legend('k=1','k=3','k=18','naive')
xlabel('Time');
ylabel('Net Raditiation error(W/m^2)');
hold off
plotACFnPACF(rad_modelC_res_k1(test3_start:test3_stop),reslag,'model C radiation test3 residue k=1');
plotACFnPACF(rad_modelC_res_k3(test3_start:test3_stop),reslag,'model C radiation test3 residue k=3');
plotACFnPACF(rad_modelC_res_k18(test3_start:test3_stop),reslag,'model C radiation test3 residue k=18');

var_test3_rad=var(test_radiation3);
fprintf('\n Prediction results of radiation test3 for model C:\n')
fprintf('\n Whiteness of prediction residue: \n')
fprintf('\n For 1-step: \n')
figure;
whitenessTest(rad_modelC_res_k1(test3_start:test3_stop))
fprintf('\n For 3-step: \n')
figure;
whitenessTest(rad_modelC_res_k3(test3_start:test3_stop))
fprintf('\n For 18-step: \n')
figure;
whitenessTest(rad_modelC_res_k18(test3_start:test3_stop))
fprintf('variance:\n')
fprintf('  The variance of the radiation test3 data is    %7.2f\n', var_test3_rad )
fprintf('  The variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test3_start:test3_stop)) )
fprintf('  The variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test3_start:test3_stop)) )
fprintf('  The variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test3_start:test3_stop)) )
fprintf('  The variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop)) )
fprintf('normalized variance:\n')
fprintf('  The normalized variance of the radiation test3 data is    %7.2f\n', 1.00)
fprintf('  The normalized variance of the 1-step prediction residual is %7.2f\n', var(rad_modelC_res_k1(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 3-step prediction residual is %7.2f\n', var(rad_modelC_res_k3(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the 18-step prediction residual is %7.2f\n', var(rad_modelC_res_k18(test3_start:test3_stop))/var_test3_rad );
fprintf('  The normalized variance of the naive prediction residual is %7.2f\n', var(naive_rad_res(test3_start:test3_stop))/var_test3_rad );
fprintf('How much is explained:\n')
fprintf('  The 1-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k1(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 3-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k3(test3_start:test3_stop))/var_test3_rad)*100 )
fprintf('  The 18-step prediction explains %3.1f%% of the variance\n',  (1-var(rad_modelC_res_k18(test3_start:test3_stop))/var_test3_rad)*100) 
