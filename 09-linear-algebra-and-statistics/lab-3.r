n<-300 
mu<-2 
sigma<-2 
x<-rnorm(n,mean=mu,sd=sigma)
N<-1000  
count<-0 
for (i in (1:N)){

    x<-rnorm(n,mean=mu,sd=sigma) 
    sigma2.hat<-sd(x)^2 
    left <- (n-1)*sigma2.hat/qchisq(0.975,df=n-1) 
    right <- (n-1)*sigma2.hat/qchisq(0.025,df=n-1)

    count<-count+as.double(left<= sigma^2 & sigma^2<=right)

}
percentage=count/N
print(percentage)