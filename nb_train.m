
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1)
numTokens = size(trainMatrix, 2)

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE

V = numTokens;
phi_k_y1 = ones(numTokens, 1);
phi_k_y0 = ones(numTokens, 1);
sample_1 = trainMatrix(find(trainCategory == 1), :);
sample_0 = trainMatrix(find(trainCategory == 0), :);
n1 = sum(sum(sample_1));
n0 = sum(sum(sample_0));
phi_y1 = log(size(sample_1, 1) / numTrainDocs);
phi_y0 = log(size(sample_0, 1) / numTrainDocs);

for t = 1 : numTokens
    phi_k_y1(t) = phi_k_y1(t) + sum(sample_1(:, t));
    phi_k_y0(t) = phi_k_y0(t) + sum(sample_0(:, t));
    phi_k_y1(t) = log(phi_k_y1(t) / (n1 + V));
    phi_k_y0(t) = log(phi_k_y0(t) / (n0 + V));
end





