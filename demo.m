clear
clc

% filename = websave('mnist_train.mat', 'https://github.com/awni/cs224n-pa4/blob/master/Simple_tSNE/mnist_train.mat?raw=true');
filename = 'mnist_train.mat';
load(filename);
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5; alg = 'svd';
map = fast_tsne(digits', numDims, pcaDims, perplexity, theta, alg);
gscatter(map(:,1), map(:,2), labels');


% show
x=map;
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

%% load validation image filenames

N = 60000;

%% create an embedding image

S = 2000; % size of full embedding image
G = zeros(S, S, 1, 'uint8');
s = 28; % size of every single image

Ntake = 60000;
for i=1:Ntake
    
    if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = uint8(reshape(digits(:,i),28,28));
    if size(I,1) ~= s || size(I,2) ~= s
        I = imresize(I, [s, s]);
    end
   
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

imshow(G);