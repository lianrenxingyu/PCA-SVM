%{
分类目标是15个人，每个人5张张图片作为测试集
%}
%%
% 选择测试集

%col_of_data = 60;%数据集一共60列，每一列代表一个人

disp('测试图片集路径:E:\MatlabProgram\作业工程\智能信息处理3.0\test');
pathname = 'E:\MatlabProgram\作业工程\智能信息处理3.0\test';
img_path_list = dir(strcat(pathname,'\*.png'));
img_num = length(img_path_list);
testdata = [];

%因为测试样本数目较少，所以直接定义了测试样本的类别RealClass，分为三类，标号1,2,3
realclass = [1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
if img_num >0
    for j = 1:img_num
        img_name = img_path_list(j).name;
        temp = imread(strcat(pathname, '/', img_name));
        temp = imresize(temp,[370,370]);
        temp = double(temp(:));
        testdata = [testdata, temp];
    end
end
col_of_test = size(testdata,2);

%中心化，归一化，将测试图片归一化,就是图片的减去平均值
meandata = mean(testdata,2);
for i = 1:size(testdata,2)
    testdata(:,i) = testdata(:,i) - meandata;
end

%%
%展示中心化之后的结果，保存为图片
figure(2)
%测试图片归一化后的结果
for i = 1:col_of_test
    test_w = reshape(testdata(:,i),370,370,3);
    subplot(4,4,i)
    imshow(test_w)
end
suptitle('测试图片归一化后的结果')
disp('提取的特征图像保存为:测试集中心化图.png');
print(2,'-dpng','测试集中心化图')

%测试样本在新坐标系的表达矩阵 p*M，
% object这时是32*45,45代表45张图片，32代表一张图片降维后的表示向量
object = W'* testdata;

% svm_test

class= multiSVM(object',multiSVMstruct,nclass);
disp('测试集的正确率为:')  
accuracy=sum(class==realclass)/length(class)
%{ 
最小距离法，寻找和待识别图片最为接近的训练图片
% 计算分类器准确率
num = 0;
for j = 1:col_of_test;
    distance = 1000000000000;
    for k = 1:col_of_data;
        % 计算欧式距离
        temp = norm(object(:,j) - reference(:,k));
        if(distance>temp)
            aimone = k;
            distance = temp;
        end
    end
    if ceil(j/5)==ceil(aimone/15)
       num = num + 1;
    end
end
accuracy = num/col_of_test
%}