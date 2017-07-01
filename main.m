%{
***************************************************************************
    
    Cheng Gong
    2017-6-24
    智能信息处理作业

    本次作业目的是对径向干扰图片、螺旋状干扰图片、麻点状干扰图片三类图片进行分类，
    最终利用的方法是PCA+SVM，即主成分分析法和支持向量机结合的方法。
    训练图片是15张，测试图片5张
***************************************************************************

%}



%main 函数是主函数
clear 
close all
tic
%%
% 批量读取指定文件夹下的图片
disp('训练图片集路径:E:\MatlabProgram\作业工程\智能信息处理3.0\train');
pathname = 'E:\MatlabProgram\作业工程\智能信息处理3.0\train';

disp('正在读取图片...');

img_path_list = dir(strcat(pathname,'\*.png'));
img_num = length(img_path_list);
imagedata = [];
if img_num >0
    for j = 1:img_num
        img_name = img_path_list(j).name;
        temp = imread(strcat(pathname, '/', img_name));
        temp = imresize(temp,[370,370]);
        temp = double(temp(:));
        imagedata = [imagedata, temp];
    end
end

fprintf('图片读取完毕。\n\n对图片进行降维处理,即PCA过程...\n\n');
%矩阵的第二维有多少数据
col_of_data = size(imagedata,2);

% 中心化 & 计算协方差矩阵
%求每一行的均值，但是每一列是一张图片的展开;
imgmean = mean(imagedata,2);
for i = 1:col_of_data
    imagedata(:,i) = imagedata(:,i) - imgmean;
end
%协方差矩阵，矩阵格式60*60
covMat = imagedata'*imagedata;
%coeff表示各个主成分的系数，latent,表示特征值，explained代表贡献率
[COEFF, latent, explained] = pcacov(covMat);

% 选择构成95%能量的特征值
i = 1;
proportion = 0;
while(proportion < 95)
    proportion = proportion + explained(i);
    i = i+1;
end
p = i - 1;%  p=32

% 特征脸，此时p=32,可以认为从60维降维成32维，减小运算量，
% 此时w可以认为是一个坐标系，然后让原始数据投影到这个坐标系上
W = imagedata*COEFF;    % N*M阶
W = W(:,1:p);           % N*p阶

% 训练样本在新座标基下的表达矩阵 p*M，
% reference这时是32*60,60代表60张图片，32代表一张图片降维后的表示向量
reference = W'*imagedata;

%%
%展示特征脸，保存为图片
figure(1)

%展示特征脸,
%p代表只取前p个特征值，利用W来展示特征脸
for i = 1:p
    train_w = reshape(W(:,i),370,370,3);
    subplot(6,6,i)
    imshow(train_w)
end
suptitle('经过PCA处理后提取的特征图像');
%保存特征图片到本地
print(1,'-dpng','特征图')
disp('提取的特征图像保存为:特征图.png');
fprintf('保存PCA两个最终结果:保存为train_pca.mat\n');
save('train_pca','W','reference')
fprintf('PCA运行结束。\n\n')

%%
%下面进行svm过程，由支持向量机进行分类
disp('开始SVM训练...');
%nclass 代表有三类
nclass = 3;
%一些训练参数，在这个简单的例子中影响不是很大

c=128;  
multiSVMstruct=multiSVMtrain( reference',nclass,c); 
disp('SVM训练完成。')
fprintf('\n')

%%
%调用测试函数，输出测试结果，并显示正确率

disp('开始测试训练结果...')
%test是测试文件名称
test
disp('测试完成。')
toc;