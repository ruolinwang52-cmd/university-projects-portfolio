function [ev,mult] = heltalsev(A,tol)
if nargin < 2 ;   
    tol = 1.e-5; end

disp('-------------------------------')
disp('Given the matrix A')
disp(A)
%Check if the elements of Matrix A are all integers.
if all(all(mod(A, 1) == 0))
        disp('All elements of the matrix are integers.');
    else
        disp('Please enter a matrix with integer elements.');
    end
%Compute eigenvalues
evs=eig(A);
%Create a variable to store rounded ev
round_ev=zeros(size(evs));
for k=1:length(evs)
    lambda=evs(k);
    %Check if the rounded value of lambda is within the preset tolerance
    if abs(lambda-round(lambda))<tol
        round_ev(k)=round(lambda);
    else
        %If not: error and exit
        disp('The matrix has a non integer eigenvalue.')
        ev=[];
        mult=[];
        return;
    end
end

%Store the distinct eigenvalues in ev
ev=unique(round_ev);
%Determine the multiplicity of each eigenvalue by checking how
%often such a value occurs in round_ev using the 'find' method.
mult=zeros(size(ev));
for k=1:length(ev)
    mult(k)=length(find(ev(k)==round_ev));
end

end


