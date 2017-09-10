function [ind, thresh] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;
best_error = inf;
% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.

for jj = 1:nn
    [X_sort, indexs] = sort(X(:, jj), 'descend');
    p_sort = p_dist(indexs);
    y_sort = y(indexs);
    % calculate possible thresholds
    % use mean between neighbourhood x
    s = X_sort(1) + 1;
    possible_thresholds = (X_sort + circshift(X_sort, 1))/2;
    possible_thresholds(1) = s;
    % change sum using changes p_sort(l) * y_sort(l)
    changes = circshift(p_sort .* y_sort, 1);
    changes(1) = 0;
    % calculate sum
    origin_sum = p_sort' * y_sort;
    sums = origin_sum * ones(mm, 1) - cumsum(changes);
    [best_low, thresh_ind] = min(sums);
    [best_high, ind_high] = max(sums);
    best_high = 1 - best_high;
    best_error_j = min(best_low, best_high);
    if best_high < best_low
        thresh_ind = ind_high;
    end
    if best_error_j < best_error
        best_error = best_error_j;
        ind = jj;
        thresh = possible_thresholds(thresh_ind);
    end
end









