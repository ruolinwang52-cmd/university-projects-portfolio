n<-10000 
x<-runif(n) 
y<-exp(-x^2) 
z_n<-cumsum(x)/(1:n) 
plot((1:n),z_n)


