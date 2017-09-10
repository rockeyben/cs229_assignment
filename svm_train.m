% Before using this method, set num_train to be the number of training
% examples you wish to read.

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('MATRIX.TRAIN.%d', num_train));

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (sparseTrainMatrix > 0);

numTrainDocs = size(Xtrain, 1);
numTokens = size(Xtrain, 2);

% Xtrain is a (numTrainDocs x numTokens) sparse matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents if the j-th token appears in
% email i.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
% vector ytrain.

% This vector should be output by this method
average_alpha = zeros(numTrainDocs, 1);

%---------------
% YOUR CODE HERE

tau = 8;
squared_Xtrain = sum(Xtrain .^ 2, 2);
gram_Xtrain = Xtrain * Xtrain';
m_train = numTrainDocs;
% full matrix requires less computing time than sparse matrix
% so here is the convertion
K = full(exp(-(repmat(squared_Xtrain, 1, m_train) + repmat(squared_Xtrain', m_train, 1) - 2 * gram_Xtrain) / (2 * tau ^2)));

max_iters =  40 * m_train;
alpha = zeros(m_train, 1);
lambda = 1 / (64 * m_train);
for iters = 1: max_iters
    i = unidrnd(m_train);
    margin = ytrain(i) * K(i, :) * alpha;
    grad = -(margin < 1) * ytrain(i) * K(:, i) + m_train * lambda * (K(:, i) * alpha(i));
    alpha = alpha - grad / sqrt(iters);
    average_alpha = average_alpha + alpha;
end

average_alpha = average_alpha / max_iters;

%---------------
