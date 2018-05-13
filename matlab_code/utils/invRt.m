function RtInv = invRt(Rt)

RtInv = [Rt(:,1:3)'  -Rt(:,1:3)'* Rt(:,4)];
