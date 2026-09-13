dat<-read.table("data.txt")
ks.test(dat$HDL[dat$SMOKING==1],"pnorm")
plot(ecdf(dat$HDL[dat$SMOKING==1]),xlim=c(0,4)) 
t<-seq(0,4,by=0.01) 
m<-mean(dat$HDL[dat$SMOKING==1],na.rm=TRUE) 
v<-var(dat$HDL[dat$SMOKING==1],na.rm=TRUE)

par(new=TRUE,col="blue")

plot(t,pnorm(t,m,sqrt(v)),xlim=c(0,4))

qqnorm(dat$HDL[dat$SMOKING==1])