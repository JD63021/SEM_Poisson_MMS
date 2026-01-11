function T = precompute_tensor_ref(p, ngauss)
% PRECOMPUTE_TENSOR_REF  Tensor-product Lagrange (equispaced) on [0,1]^2 at order p.
% Returns reference (Q x (p+1)^2) arrays in **tensor order** (i fast, j slow).
% T = precompute_tensor_ref(p, ngauss)
%  .nodes1D (N1x1), .gauss1D.nodes/weights (1xn), .Pts(Qx2), .Wts(Qx1)
%  .Phi (QxN2), .dPhi_dxi (QxN2), .dPhi_deta (QxN2), .IJ (N2x2)

N1 = p+1; N2 = N1*N1;

% 1D nodes & bary weights
r1 = linspace(0,1,N1).';
w1 = bary_weights(r1);

% Gauss–Legendre on [0,1]
[xg, wg] = gauss_legendre_on01(ngauss);
[XI,ETA] = meshgrid(xg, xg);
[WX,WY]  = meshgrid(wg, wg);
Pts = [XI(:), ETA(:)];
Wts = (WX(:).*WY(:));
Q   = numel(Wts);

% 1D evals/derivs at Gauss nodes
[Lx, dLx] = bary_eval_and_deriv_matrix(r1, w1, xg);
[Ly, dLy] = bary_eval_and_deriv_matrix(r1, w1, xg);

% tensor enumeration
IJ = zeros(N2,2); m=0;
for j=1:N1, for i=1:N1, m=m+1; IJ(m,:)=[i,j]; end, end

% build arrays
Phi      = zeros(Q,N2);
dPhi_dxi = zeros(Q,N2);
dPhi_deta= zeros(Q,N2);
for m = 1:N2
    i = IJ(m,1); j = IJ(m,2);
    Z    = Ly(j,:).' * Lx(i,:);
    Zxi  = Ly(j,:).' * dLx(i,:);
    Zeta = dLy(j,:).' * Lx(i,:);
    Phi(:,m)      = Z(:);
    dPhi_dxi(:,m) = Zxi(:);
    dPhi_deta(:,m)= Zeta(:);
end

% pack
T = struct('p',p,'N1',N1,'N2',N2, ...
           'nodes1D',r1,'gauss1D',struct('nodes',xg(:).','weights',wg(:).'), ...
           'Pts',Pts,'Wts',Wts, ...
           'Phi',Phi,'dPhi_dxi',dPhi_dxi,'dPhi_deta',dPhi_deta, ...
           'IJ',IJ);
end

function [x,w] = gauss_legendre_on01(n)
if n<=0, x=[]; w=[]; return; end
beta=0.5./sqrt(1 - (2*(1:n-1)).^(-2));
J = diag(beta,1)+diag(beta,-1);
[V,D] = eig(J);
t = diag(D); [t,perm]=sort(t); V=V(:,perm);
w = 2*(V(1,:).^2).';
x = (t+1)/2; w = w/2;
end
