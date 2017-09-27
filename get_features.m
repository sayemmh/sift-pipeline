% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points.

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)


% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature vector should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

function [features] = get_features(image, x, y, feature_width, sigma, threshold)

  features = zeros(size(x,1),128);
  small = fspecial('gaussian',sigma,sigma);
  I = imfilter(image,small);
  sobel = fspecial('sobel');
  sobelt = sobel';
  Ix_s = imfilter(imfilter(I, sobelt), fspecial('gaussian',2.*sigma,2.*sigma));
  Iy_s = imfilter(imfilter(I, sobel), fspecial('gaussian',2.*sigma,2.*sigma));
  atanIyIx = ceil(4.*(atan2(Iy_s,Ix_s)./pi+1));
  Ix2y2 = sqrt((Iy_s.^2)+(Ix_s.^2));

  x_len = size(x);
  for n = 1:x_len
      xi = x(n);
      yi = y(n);
      F = zeros(1, 128);
      atan_sub = atanIyIx(yi-10:yi+10, xi-10:xi+10);
      Ix2y2_sub = Ix2y2(yi-10:yi+10, xi-10:xi+10);

      for i = 0:3
          for j = 0:3
              p1 = atan_sub(4.*i+1:4.*i+4, 4.*j+1:4.*j+4);
              p2 = Ix2y2_sub(4.*i+1:4.*i+4, 4.*j+1:4.*j+4);
              for x1 = 1:8
                  pxl = sum(p2(p1==x1));
                  F(1,8.*(4.*i+j)+x1) = pxl;
              end
          end
      end

      % You do not need to do the normalize -> threshold -> normalize again
      % operation as detailed in Szeliski and the SIFT paper. It can help, though.

      oneNormF = 1./norm(F, 1);
      F = F.*oneNormF;

      % TODO: Add into function params.
      % threshold = 0.2;

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.
      F(F > threshold) = threshold;
      oneNormF = 1./norm(F, 1);
      features(n,:) = F.*oneNormF;
  end
end
