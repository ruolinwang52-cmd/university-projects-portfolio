function J = jordanmatris(A,tol)
if nargin < 2 ;
    tol = 1.e-5; end
[ev,mult] = heltalsev(A,tol);

%if heltalsev found a non integer value then ev=[], mult=[];
if isempty(ev)
    disp('Try another matrix')
    J=[];
    return;
end

%Determine size of the matrix
n=size(A,1);
J=[];
%loop through all ev
for k=1:length(ev)
    lambda=ev(k);
    B=A-lambda*eye(n);
    %Store powers of B from 1
    PowerB=B;
    %We will now do a while loop. When completed the variable done
    %will change from 0 to 1.
    done=0;
    ind=1;
    r=[];
    while ~done
        nullPowerB=null(PowerB);
        dimnullPowerB=size(nullPowerB,2);
        if dimnullPowerB==0
            error('Empty null space')
        end
        r(ind)=dimnullPowerB; %r(ind)Should be dimension null space of B^ind
        if ind>1 
            if r(ind)==r(ind-1)
                disp('Error,Please enter a new matrix')
                return;
            end
        end
        %If we have reached the multiplicity then we are done 
        if r(ind)==mult(k)
            done=1;
        else
            ind=ind+1;
            PowerB=PowerB*B;
        end
    end
   
    N=length(r);
    %Now r(1)<r(2)<....<r(N) is a sequence of dim(null space)

    %Determine s(j) which is the number of Jordan blocks of size
    %at least j by j for the eigenvalue lambda, j=1...N
    s(1)=r(1);
    for j=2:N
        s(j)=r(j)-r(j-1);
    end
    %Determine m(j) which is the number of Jordan blocks of size
    %exactly j by j for the eigenvalue lambda.
    m(N)=s(N);
    for j=N-1:-1:1
        m(j)=s(j)-s(j+1);
    end
    %Create Jordan blocks for the eigenvalue lambda to matrix J.
    for j=1:N
        for i=1:m(j)
            %make a j by j Jordan block
            jordanblock=lambda*eye(j)+diag(ones(j-1,1),1);
            %add it to matrix J
            J=blkdiag(J,jordanblock);
        end
    end
end

end


