
%%
function r= cauchyrnd(varargin)

% USAGE:       r= cauchyrnd(a, b, n, ...)
% 
% Generate random numbers from the Cauchy distribution, r= a + b*tan(pi*(rand(n)-0.5)).
% 
% ARGUMENTS:
% a (default value: 0.0) must be scalars or size(x).
% b (b>0, default value: 1.0) must be scalars or size(x).
% n and onwards (default value: 1) specifies the dimension of the output.
% 
% EXAMPLE:
% r= cauchyrnd(0, 1, 10); % A 10 by 10 array of random values, Cauchy distributed.
% 
% SEE ALSO:    cauchycdf, cauchyfit, cauchyinv, cauchypdf.
% 
% Copyright (C) Peder Axensten <peder at axensten dot se>
% 
% HISTORY:
% Version 1.0, 2006-07-10.
% Version 1.1, 2006-07-26.
% - Added cauchyfit to the cauchy package. 
% Version 1.2, 2006-07-31:
% - cauchyinv(0, ...) returned a large negative number but should be -Inf. 
% - Size comparison in argument check didn't work. 
% - Various other improvements to check list. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Default values
	a=	0.0;
	b=	1.0;
	n=	1;
	
	
	% Check the arguments
	if(nargin >= 1)
		a=	varargin{1};
		if(nargin >= 2)
			b=			varargin{2};
			b(b <= 0)=	NaN;	% Make NaN of out of range values.
			if(nargin >= 3),	n=	[varargin{3:end}];		end
		end
	end
	
	
	% Generate
	r=	cauchyinv(rand(n), a, b);
end

function x= cauchyinv(p, varargin)

% USAGE:       x= cauchyinv(p, a, b)
% 
% Inverse of the Cauchy cumulative distribution function (cdf), x= a + b*tan(pi*(p-0.5)).
% 
% ARGUMENTS:
% p (0<=p<=1) might be of any dimension.
% a (default value: 0.0) must be scalars or size(p).
% b (b>0, default value: 1.0) must be scalars or size(p).
% 
% EXAMPLE:
% p= 0:0.01:1;
% plot(cauchyinv(p), p);
% 
% SEE ALSO:    cauchycdf, cauchyfit, cauchypdf, cauchyrnd.
% 
% Copyright (C) Peder Axensten <peder at axensten dot se>
% 
% HISTORY:
% Version 1.0, 2006-07-10.
% Version 1.1, 2006-07-26.
% - Added cauchyfit to the cauchy package. 
% Version 1.2, 2006-07-31:
% - cauchyinv(0, ...) returned a large negative number but should be -Inf. 
% - Size comparison in argument check didn't work. 
% - Various other improvements to check list. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Default values
	a=	0.0;
	b=	1.0;
	
	
	% Check the arguments
	if(nargin >= 2)
		a=	varargin{1};
		if(nargin == 3)
			b=			varargin{2};
			b(b <= 0)=	NaN;	% Make NaN of out of range values.
		end
	end
	if((nargin < 1) || (nargin > 3))
		error('At least one argument, at most three!');
	end
	
	p(p < 0 | 1 < p)=	NaN;
	
	
	% Calculate
	x=			a + b.*tan(pi*(p-0.5));
	
	% Extreme values. 
	if(numel(p) == 1), 	p= repmat(p, size(x));		end
	x(p == 0)=	-Inf;
	x(p == 1)=	Inf;
end

