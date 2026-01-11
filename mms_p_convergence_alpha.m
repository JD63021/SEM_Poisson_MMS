% mms_p_convergence_alpha.m
% Exponential p-convergence for Poisson: u = sin(kw x) sin(kw y)
% Produces the "alpha vs p" graph: local and average exponential rates.

clear; clc;

% --- knobs ---
base   = 'squaretest'; % expects squaretest1.m ... squaretest10.m
p_list = 1:10;
kw     = 1;            % wave number (analytic & smooth)
minGL  = 10;           % >=10×10 GL always (safe)

fprintf('=== MMS Poisson p-convergence (kw=%d) ===\n', kw);

E = nan(size(p_list));     % L2 errors
H = nan(size(p_list));     % element size (info only)

for t = 1:numel(p_list)
    p = p_list(t);
    meshFile = sprintf('%s%d.m', base, p);
    ngauss   = max(minGL, p+6);

    % ---------- read mesh ----------
    [nodes, elems, bcs] = meshreader(meshFile, p, []);
    n    = nodes.N;  X = nodes.x;  Y = nodes.y;
    nloc = elems.nloc;  NE = elems.NE;

    % ---------- reference shapes (tensor order) ----------
    quad  = ref_quad_cache(p, ngauss);
    PhiT  = quad.Phi_ref.';      % nloc x Q
    DXIT  = quad.dPhi_dxi.';     % nloc x Q
    DETAT = quad.dPhi_deta.';    % nloc x Q
    gw    = quad.Wts(:);         % Q x 1
    Q     = numel(gw);

    % 1D nodes/weights for alignment
    N1 = p+1;  r1 = linspace(0,1,N1).';  w1 = bary_weights(r1);
    IJ = zeros(nloc,2); c=0; for j=1:N1, for i=1:N1, c=c+1; IJ(c,:)=[i,j]; end, end

    % MMS definitions
    u_exact = @(x,y) sin(kw*x).*sin(kw*y);
    f_fun   = @(x,y) (kw^2 + kw^2) .* u_exact(x,y);  % -Δu = (kx^2+ky^2) u

    % ---------- assemble ----------
    K = spalloc(n,n, NE*nloc*10);
    F = zeros(n,1);
    h_acc = zeros(NE,1);

    for e = 1:NE
        ids = elems.quads(e,:).';
        xe = nodes.x(ids);  ye = nodes.y(ids);

        xmin=min(xe); xmax=max(xe); dx=xmax-xmin;
        ymin=min(ye); ymax=max(ye); dy=ymax-ymin;
        J = dx*dy;

        % per-element gmsh->tensor alignment (robust)
        xi_n  = (xe - xmin)/dx;           % nloc x 1
        eta_n = (ye - ymin)/dy;           % nloc x 1
        Lx = bary_eval_matrix(r1, w1, xi_n.');   % N1 x nloc
        Ly = bary_eval_matrix(r1, w1, eta_n.');  % N1 x nloc
        V = zeros(nloc,nloc);
        for m=1:nloc
            i = IJ(m,1); j = IJ(m,2);
            V(:,m) = (Lx(i,:).').*(Ly(j,:).');
        end
        [~, rows] = max(V,[],1);            % rows(m): gmsh row where tensor m is
        perm_g2t = zeros(nloc,1); for m=1:nloc, perm_g2t(rows(m)) = m; end

        % take rows in gmsh local order
        Phi  = PhiT(perm_g2t,:);           % nloc x Q
        Dxi  = DXIT(perm_g2t,:);           % nloc x Q
        Deta = DETAT(perm_g2t,:);          % nloc x Q

        % map GP to physical & scale derivs
        xq = Phi.'*xe;  yq = Phi.'*ye;     % Q x 1
        dphidx = (1/dx) * Dxi;
        dphidy = (1/dy) * Deta;
        W = gw * J;

        % element matrices
        Ke = zeros(nloc,nloc);
        fe = zeros(nloc,1);
        fq = f_fun(xq,yq);
        for g = 1:Q
            v  = Phi(:,g);
            gx = dphidx(:,g);
            gy = dphidy(:,g);
            w  = W(g);
            Ke = Ke + w*(gx*gx.' + gy*gy.');
            fe = fe + w*(v * fq(g));
        end

        K(ids,ids) = K(ids,ids) + Ke;
        F(ids)     = F(ids)     + fe;

        h_acc(e) = sqrt(dx*dy);
    end

    % Dirichlet BCs
    bc = bcs.allNodes;
    K(bc,:) = 0; K(sub2ind(size(K), bc, bc)) = 1; F(bc) = 0;

    uh = K\F;

    % ---------- L2 error ----------
    L2sq = 0;
    for e = 1:NE
        ids = elems.quads(e,:).';
        xe = nodes.x(ids);  ye = nodes.y(ids);

        xmin=min(xe); xmax=max(xe); dx=xmax-xmin;
        ymin=min(ye); ymax=max(ye); dy=ymax-ymin;
        J = dx*dy;

        xi_n  = (xe - xmin)/dx;  eta_n = (ye - ymin)/dy;
        Lx = bary_eval_matrix(r1, w1, xi_n.');
        Ly = bary_eval_matrix(r1, w1, eta_n.');
        V = zeros(nloc,nloc);
        for m=1:nloc
            i = IJ(m,1); j = IJ(m,2);
            V(:,m) = (Lx(i,:).').*(Ly(j,:).');
        end
        [~, rows] = max(V,[],1);
        perm_g2t = zeros(nloc,1); for m=1:nloc, perm_g2t(rows(m)) = m; end

        Phi = PhiT(perm_g2t,:);           % nloc x Q
        xq  = Phi.'*xe;  yq = Phi.'*ye;
        uhq = Phi.'*uh(ids);
        uq  = u_exact(xq,yq);
        L2sq = L2sq + sum( (uhq - uq).^2 .* (gw*J) );
    end

    E(t) = sqrt(L2sq);
    H(t) = mean(h_acc);

    fprintf('p=%2d | nloc=%3d | Q=%d | ln(E)=%.3e\n', p, nloc, Q, log(E(t)));
end

% ---------- exponential scaling curves ----------
lnE = log(E(:));
pvec = p_list(:);

% Local (pairwise) rate: alpha_local(p) = ln(E_{p-1}) - ln(E_p), p>=2
alpha_local = nan(size(pvec));
alpha_local(2:end) = lnE(1:end-1) - lnE(2:end);

% Global/average rate: alpha_avg(p) = - ln(E_p)/p
alpha_avg = - lnE ./ pvec;

% Base-10 versions (if you want 10^{-beta p} picture)
log10E = log10(E(:));
beta_local = nan(size(pvec)); beta_local(2:end) = log10E(1:end-1) - log10E(2:end);
beta_avg   = - log10E ./ pvec;

% Fit a single asymptotic alpha using last few points (optional)
tail = max(4, ceil(numel(pvec)/3));
coef = polyfit(pvec(end-tail+1:end), lnE(end-tail+1:end), 1);  % lnE = a*p + b
alpha_fit = -coef(1);

fprintf('\nAsymptotic alpha (fit on last %d p''s) ≈ %.6f (natural log base)\n', tail, alpha_fit);

% ---------- plots ----------
figure;
plot(pvec, alpha_local, 'o-', 'LineWidth', 1.5); hold on;
plot(pvec, alpha_avg,   's--', 'LineWidth', 1.5);
yline(alpha_fit, ':', 'LineWidth', 1.2);
grid on; xlabel('Polynomial order p'); ylabel('\alpha (natural log base)');
title(sprintf('Exponential scaling vs p (kw=%d): local & average', kw));
legend('\alpha_{local}(p) = ln(E_{p-1})-ln(E_p)', '\alpha_{avg}(p) = -ln(E_p)/p', ...
       sprintf('\\alpha_{fit} \\approx %.3f', alpha_fit), 'Location','southeast');

% Bonus: the classic semilogy E vs p beside the scaling curves
figure;
semilogy(pvec, E, 'o-', 'LineWidth', 1.5); grid on;
xlabel('p'); ylabel('L_2 error'); title('L_2 error vs p (semilogy)');

% ===== helpers =====
function quad = ref_quad_cache(p, ngauss)
    N1 = p+1; N2 = N1*N1;
    r1 = linspace(0,1,N1).';
    w1 = bary_weights(r1);

    [xg, wg] = gauss_legendre_on01(ngauss);
    [XI,ETA] = meshgrid(xg, xg);
    [WX,WY]  = meshgrid(wg, wg);
    Pts = [XI(:), ETA(:)];
    Wts = (WX(:).*WY(:)); Q = numel(Wts);

    [Lx, dLx] = bary_eval_and_deriv_matrix(r1, w1, xg);
    [Ly, dLy] = bary_eval_and_deriv_matrix(r1, w1, xg);

    Phi_ref   = zeros(Q, N2);
    dPhi_dxi  = zeros(Q, N2);
    dPhi_deta = zeros(Q, N2);

    IJ = zeros(N2,2); m = 0;
    for j=1:N1, for i=1:N1, m=m+1; IJ(m,:)=[i,j]; end, end

    for m = 1:N2
        i = IJ(m,1); j = IJ(m,2);
        Z      = Ly(j,:).' * Lx(i,:);   % n x n
        Zxi    = Ly(j,:).' * dLx(i,:);
        Zeta   = dLy(j,:).' * Lx(i,:);
        Phi_ref(:,m)   = Z(:);
        dPhi_dxi(:,m)  = Zxi(:);
        dPhi_deta(:,m) = Zeta(:);
    end

    quad = struct('p',p,'ngauss',ngauss,'nodes1D',r1, ...
                  'Pts',Pts,'Wts',Wts, ...
                  'Phi_ref',Phi_ref,'dPhi_dxi',dPhi_dxi,'dPhi_deta',dPhi_deta);
end

function w = bary_weights(x)
    x = x(:); n=numel(x); w=ones(n,1);
    for j=1:n, for k=[1:j-1, j+1:n], w(j)=w(j)*(x(j)-x(k)); end, end
    w = 1./w;
end

function L = bary_eval_matrix(xnodes, w, xq)
    xn = xnodes(:); xq = xq(:).'; L = zeros(numel(xn), numel(xq));
    for k=1:numel(xq)
        x = xq(k); diffs = x - xn;
        hit = find(abs(diffs) < 1e-14, 1);
        if ~isempty(hit)
            e=zeros(numel(xn),1); e(hit)=1; L(:,k)=e;
        else
            t = w./diffs; L(:,k) = t / sum(t);
        end
    end
end

function [L, dL] = bary_eval_and_deriv_matrix(xnodes,w,xq)
    xn=xnodes(:); xq=xq(:).'; n=numel(xnodes); m=numel(xq);
    L=zeros(n,m); dL=zeros(n,m);
    for k=1:m
        x=xq(k); diffs=x-xn; hit=find(abs(diffs)<1e-14,1);
        if ~isempty(hit)
            k0=hit; e=zeros(n,1); e(k0)=1; L(:,k)=e;
            % derivative at node: standard barycentric identities
            for i=1:n
                if i==k0
                    s=0; for j=1:n, if j~=i, s = s - w(j)/(w(i)*(xn(i)-xn(j))); end, end
                    dL(i,k)=s;
                else
                    dL(i,k)= w(i)/(w(k0)*(xn(k0)-xn(i)));
                end
            end
        else
            t = w./diffs;   T = sum(t);
            s = -w./(diffs.^2); S = sum(s);
            L(:,k)  = t / T;
            dL(:,k) = (s*T - t*S) / (T*T);
        end
    end
end

function [x,w] = gauss_legendre_on01(n)
    if n<=0, x=[]; w=[]; return; end
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    J = diag(beta,1) + diag(beta,-1);
    [V,D] = eig(J);
    t = diag(D); [t,perm] = sort(t); V = V(:,perm);
    w = 2*(V(1,:).^2).'; x = (t+1)/2; w = w/2;
end
