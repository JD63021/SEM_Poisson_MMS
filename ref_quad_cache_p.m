function quad = ref_quad_cache_p(p, varargin)
% REF_QUAD_CACHE_P  Precompute tensor Lagrange shapes (order p) on [0,1]^2
% at an n×n Gauss–Legendre grid.
%
% quad fields (tensor column order: i=1..p+1 fast, j=1..p+1 slow):
%   .p, .ngauss, .nodes1D (p+1 x 1)
%   .Pts   (Q x 2)  GL tensor points on [0,1]^2
%   .Wts   (Q x 1)  tensor weights
%   .Phi_ref   (Q x (p+1)^2)
%   .dPhi_dxi  (Q x (p+1)^2)
%   .dPhi_deta (Q x (p+1)^2)

    % options
    ngauss = 6;
    for k=1:2:numel(varargin)
        switch lower(varargin{k})
            case 'ngauss', ngauss = varargin{k+1};
            otherwise, error('Unknown option "%s"', varargin{k});
        end
    end
    N1 = p+1; N2 = N1*N1;

    % 1D nodes (equispaced) & barycentric weights on [0,1]
    r1 = linspace(0,1,N1).';  w1 = bary_weights(r1);

    % n-point GL on [0,1] and tensor grid
    [xg, wg] = gauss_legendre_on01(ngauss);
    [XI,ETA] = meshgrid(xg, xg);
    [WX,WY]  = meshgrid(wg, wg);
    Pts = [XI(:), ETA(:)];    % Q x 2
    Wts = WX(:).*WY(:);       % Q x 1
    Q   = numel(Wts);

    % 1D values & derivs at GL nodes
    [Lx, dLx] = bary_eval_and_deriv_matrix(r1, w1, xg);  % N1 x ngauss
    [Ly, dLy] = bary_eval_and_deriv_matrix(r1, w1, xg);

    % enumerate tensor basis (i fast, j slow)
    IJ = zeros(N2,2); m = 0;
    for j=1:N1, for i=1:N1, m=m+1; IJ(m,:)=[i,j]; end, end

    % build Q x N2 arrays
    Phi_ref   = zeros(Q, N2);
    dPhi_dxi  = zeros(Q, N2);
    dPhi_deta = zeros(Q, N2);

    % outer products on (ngauss x ngauss), then vectorize
    for m=1:N2
        i = IJ(m,1); j = IJ(m,2);
        Z    = Ly(j,:).' * Lx(i,:);     % values
        Zxi  = Ly(j,:).' * dLx(i,:);    % ∂/∂xi
        Zeta = dLy(j,:).' * Lx(i,:);    % ∂/∂eta
        Phi_ref(:,m)   = Z(:);
        dPhi_dxi(:,m)  = Zxi(:);
        dPhi_deta(:,m) = Zeta(:);
    end

    quad = struct('p',p,'ngauss',ngauss,'nodes1D',r1,'Pts',Pts,'Wts',Wts, ...
                  'Phi_ref',Phi_ref,'dPhi_dxi',dPhi_dxi,'dPhi_deta',dPhi_deta);
end

% ===== helpers =====
function w = bary_weights(x)
    x = x(:); n=numel(x); w=ones(n,1);
    for j=1:n, for k=[1:j-1, j+1:n], w(j)=w(j)*(x(j)-x(k)); end, end
    w = 1./w;
end

function [L, dL] = bary_eval_and_deriv_matrix(xnodes,w,xq)
    xn = xnodes(:); xq = xq(:).'; n=numel(xn); m=numel(xq);
    L  = zeros(n,m); dL = zeros(n,m);
    for k=1:m
        x = xq(k); diffs = x - xn; hit = find(abs(diffs)<1e-14,1);
        if ~isempty(hit)
            k0 = hit; e=zeros(n,1); e(k0)=1; L(:,k)=e;
            % derivative via barycentric identity at node
            s = 0;
            for j=[1:k0-1,k0+1:n]
                s = s - w(j)/(w(k0)*(xn(k0)-xn(j)));
            end
            dL(k0,k) = s;
            for i=[1:k0-1,k0+1:n]
                dL(i,k) = w(i)/(w(k0)*(xn(k0)-xn(i)));
            end
        else
            t = w./diffs;  T = sum(t);
            s = -w./(diffs.^2); S = sum(s);
            L(:,k)  = t/T;
            dL(:,k) = (s*T - t*S)/(T*T);
        end
    end
end

function [x,w] = gauss_legendre_on01(n)
    if n<=0, x=[]; w=[]; return; end
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    J = diag(beta,1) + diag(beta,-1);
    [V,D] = eig(J); t = diag(D); [t,perm] = sort(t); V = V(:,perm);
    w = 2*(V(1,:).^2).'; x = (t+1)/2; w = w/2;
end
