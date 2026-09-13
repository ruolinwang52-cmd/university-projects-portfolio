x<-runif(1000)
y<-as.double(x>0.4)
y1<-sum(y)/1000
y0<-1-y1
plot(c(0,1),c(y0,y1),type="p",xlim=c(0,1),ylim=c(0,1))