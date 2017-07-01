function [ multiSVMstruct ] =multiSVMtrain( traindata,nclass,c)  
%多类别的SVM训练器 
% 选取三类图片中的两类进行svm训练，最后保存训练返回的结果svmstruct组成一个矩阵，返回结果

%下面的15数值代表，训练图片15张，更改训练张数的时候一定要记得修改这个参数，
%不想更改函数输入参数了，把15写进函数输入参数更好
for i=1:nclass-1  
    for j=i+1:nclass          
        X=[traindata(15*(i-1)+1:15*i,:);traindata(15*(j-1)+1:15*j,:)];
        Y=[ones(15,1);zeros(15,1)];  
      %简单的选取rbf作为分类的核函数
       multiSVMstruct{i}{j}=svmtrain(X,Y,'Kernel_Function','rbf','boxconstraint',c); 
      end  
end  
end
