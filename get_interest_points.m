% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns alpha set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (alpha) suppress boundary interest
%   points (where alpha feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.

% Implement the Harris C detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are alpha bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run alpha max() operator on each sliding window. You
% could use this to ensure that every interest point is at alpha local maximum
% of cornerness.


function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width, alpha, threshold)
  % 1) Compute horizontal and vertical derivatives Ix and Iy by convolving original image with derivatives of Gaussians
  % 2) Compute three images corresponding to the outer products of these gradients
  % 3) Convolve each of these images with alpha larger Gaussian
  % 4) Compute alpha scalar interest measure using one of the formulas discussed above
  % 5) Find local maxima above alpha certain threshold and report them as detected feature point locations
  [rows, cols, layers] = size(image);
  sigma = 2;
  g_filter = fspecial('gaussian', 2.*sigma+1, sigma+1);
  g_filter2 = fspecial('gaussian', sigma+1, sigma);

  % x and y first derivative filters.
  Ix_f = imfilter(g_filter, [1,-1], 'symmetric');
  Iy_f = imfilter(g_filter, [1;-1], 'symmetric');

  Ixxf = zeros(rows, cols, layers);
  Iyyf = zeros(rows, cols, layers);
  Ix2 = zeros(rows, cols, layers);
  Iy2 = zeros(rows, cols, layers);
  IxIy = zeros(rows, cols, layers);

  for i = 1:layers
      Ixxf(:,:,i) = imfilter(image(:,:,i), Ix_f);
      Iyyf(:,:,i) = imfilter(image(:,:,i), Iy_f);

      Ix2(:,:,i) = imfilter(Ixxf(:,:,i) .^ 2, g_filter2);
      Iy2(:,:,i) = imfilter(Iyyf(:,:,i) .^ 2, g_filter2);
      IxIy(:,:,i) = imfilter(Ixxf(:,:,i) .* Iyyf(:,:,i), g_filter2);
  end

  % TODO: search for good alpha and good threshold
  % alpha = 0.065; threshold = 0.015;
  % Ix2 Iy2 - (IxIy)^2
  C = Ix2 .* Iy2 - IxIy .^2 - alpha * (Ix2 + Iy2) .^2;
  % clear
  C(1:feature_width,:,:) = 0;
  C(end-feature_width:end,:,:) = 0;
  C(:,1:feature_width,:) = 0;
  C(:,end-feature_width:end,:) = 0;

  threshold = threshold .* max(max(C));
  C = C .* (C > threshold);

  C = C.*(C == colfilt(C,[3 3],'sliding',@max));
  [y, x] = find(C);

end
