


	A= zeros(3);
    J=jordanmatris(A)

	A = eye(4);  
    J=jordanmatris(A)

	A = [1 1 1; 1 1 1; -2 -2 -2]; 
    J=jordanmatris(A)


	A =[-9  11 -21    63 -252;
        70 -69 141  -421 1684;
      -575 575 -1149 3451 -13801;
       3891 -3891 7782 -23345 93365;
       1024 -1024 2048 -6144 24572]; 
    J=jordanmatris(A)



	A = compan(poly(1:10));
    J=jordanmatris(A)


    A = [3 -4;4 3];
    J=jordanmatris(A)
    


	A = diag([1.000001 1]);
    J=jordanmatris(A)
