function bp_train
x=ones(25,2);
x(:,1)=0;
t=zeros(4,200);
t(1,141:160)=1;
t(1,161:180)=1;
t(1,181:200)=1;
t(2,61:80)=1;
t(2,81:100)=1;
t(2:3,101:120)=1;
t(2:4,121:140)=1;
t(3,21:40)=1;
t(3:4,41:60)=1;
t(3,181:200)=1;
t(4,1:20)=1;
t(4,81:100)=1;
t(4,161:180)=1;

[f1,f2,f3,f4,class]=textread('iris_traindata.txt','%f %f %f %f %f ',90);

[input,minI,maxI]=premnmx([f1,f2,f3,f4]'); 

s=length(class);
output=zeros(s,3);
for i=1:s
    output(i,class(i))=1;
end

net=newff(minmax(input),[10 3],{'logsig' 'purelin'},'traingdx');
net.trainparam.show=50;
net.trainparam.epochs=500;
net.trainparam.goal=0.01;
net.trainparam.lr=0.01;

net=train(net,input,output');


[t1,t2,t3,t4,c]=textread('iris_testdata.txt','%f %f %f %f %f ',60);
testInput=tramnmx([t1,t2,t3,t4]',minI,maxI);
Y=sim(net,testInput);

[s1,s2]=size(Y);
hitNum=0;
for i=1:s2
    [m,Index]=max(Y(:,i));
    if(Index==c(i))
        hitNum=hitNum+1;
    end
    
end
    
sprintf('shi bie lv shi %3.3f%%',100*hitNum/s2);




