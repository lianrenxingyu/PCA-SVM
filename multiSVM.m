%%  
function [ class] = multiSVM(testface,multiSVMstruct,nclass)  
%对测试数据进行分类
%由于支持向量机是只能区分两类的，所以需要进行下面的处理，让支持向量机可以完成多类图片的区分
m=size(testface,1);  
voting=zeros(m,nclass);  

%% 两两投票，累加结果，最后选择最大的数值作为投票结果
for i=1:nclass-1  
    for j=i+1:nclass  
        class=svmclassify(multiSVMstruct{i}{j},testface);  
        voting(:,i)=voting(:,i)+(class==1);  
        voting(:,j)=voting(:,j)+(class==0);  
    end  
end  
[~,class]=max(voting,[],2);
end 