% ns_mms_Q10Q9_picard.m
% Q10 (velocity) – Q9 (pressure) MMS with *convection ON*, Picard linearization
% Predictor = exact MMS velocity at the same GL points (so one solve is enough)
clear; clc;

%% knobs
meshVel = 'squareS10.m';   % velocity mesh (Q10)
meshPres= 'squareS9.m';    % pressure mesh (Q9)
mu  = 1;                   % viscosity
Re  = 1;                   % mild convection (set 0 to recover Stokes)
ngauss_v = 12;             % GL x GL for velocity integrals
ngauss_p = 12;             % GL x GL for pressure integrals (safe)
show_plots = true;

%% load meshes
[nV, eV, bV] = meshreader(meshVel, 10, []);
[nP, eP, bP] = meshreader(meshPres,  9, []);
NV = nV.N; NP = nP.N;

%% reference caches (tensor order)
Q10 = ref_quad_cache_p10('ngauss', ngauss_v);  % already in your folder
Q9  = ref_quad_cache_p9( 'ngauss', ngauss_p);  % if you don't have this, reuse p10 with p=9 analogue

PhiVt  = Q10.Phi_ref.';   % (NlocV x Qv)
DXIVt  = Q10.dPhi_dxi.';  % (NlocV x Qv)
DETAVt = Q10.dPhi_deta.'; % (NlocV x Qv)
wV     = Q10.Wts(:);      % (Qv x 1)
Qv     = numel(wV);
NlocV  = eV.nloc;         % =121 for p=10

% small helper for gmsh->tensor alignment per element (robust)
nodes1D = Q10.nodes1D;  w1 = bary_weights(nodes1D);
IJ = zeros(NlocV,2); c=0; for j=1:numel(nodes1D), for i=1:numel(nodes1D), c=c+1; IJ(c,:)=[i,j]; end, end

%% MMS fields (exact velocity, pressure) on [0,1]^2
u_ex = @(x,y)  2*pi*(sin(pi*x).^2).*sin(pi*y).*cos(pi*y);
v_ex = @(x,y) -2*pi*(sin(pi*y).^2).*sin(pi*x).*cos(pi*x);
p_ex = @(x,y)  sin(pi*x).*sin(pi*y);

% Forced, nonlinear momentum (your formulas, marked “for non linear problem”)
fu_ex = @(x,y) 2*pi^3*sin(2*pi*x) + pi*cos(pi*x).*sin(pi*y) + 6*mu*pi^3*sin(2*pi*y) ...
              - 4*pi^3*cos(pi*x).^3.*sin(pi*x) ...
              + 4*pi^3*cos(pi*x).^3.*cos(pi*y).^2.*sin(pi*x) ...
              - 4*pi^3*cos(pi*x).*cos(pi*y).^2.*sin(pi*x) ...
              - 16*mu*pi^3*cos(pi*x).^2.*cos(pi*y).*sin(pi*y);

fv_ex = @(x,y) 2*pi^3*sin(2*pi*y) + pi*cos(pi*y).*sin(pi*x) - 6*mu*pi^3*sin(2*pi*x) ...
              - 4*pi^3*cos(pi*y).^3.*sin(pi*y) ...
              + 4*pi^3*cos(pi*x).^2.*cos(pi*y).^3.*sin(pi*y) ...
              - 4*pi^3*cos(pi*x).^2.*cos(pi*y).*sin(pi*y) ...
              + 16*mu*pi^3*cos(pi*x).*cos(pi*y).^2.*sin(pi*x);

%% Build Picard predictor a = (au,av) at element Gauss points (same rule as assembly)
% We also collect per-element gmsh->tensor permutations
au_cell = cell(eV.NE,1);
av_cell = cell(eV.NE,1);

for e = 1:eV.NE
    ids = eV.quads(e,:).';
    xe  = nV.x(ids);  ye  = nV.y(ids);

    xmin=min(xe); xmax=max(xe); dx=xmax-xmin;
    ymin=min(ye); ymax=max(ye); dy=ymax-ymin;

    % gmsh->tensor perm via barycentric match
    xi_n  = (xe - xmin)/max(dx,eps);
    eta_n = (ye - ymin)/max(dy,eps);
    Lx = bary_eval_matrix(nodes1D, w1, xi_n.');
    Ly = bary_eval_matrix(nodes1D, w1, eta_n.');
    V  = zeros(NlocV,NlocV);
    for m=1:NlocV
        i = IJ(m,1); j = IJ(m,2);
        V(:,m) = (Lx(i,:).').*(Ly(j,:).');
    end
    [~, rows] = max(V,[],1);
    perm_g2t = zeros(NlocV,1); for m=1:NlocV, perm_g2t(rows(m)) = m; end

    % take rows in Gmsh order
    PhiV = PhiVt(perm_g2t,:);   % NlocV x Qv

    % GP in physical coords
    xq = PhiV.' * xe;  yq = PhiV.' * ye;   % (Qv x 1)

    % predictor velocities at GPs
    au_cell{e} = u_ex(xq,yq);
    av_cell{e} = v_ex(xq,yq);
end

%% Assemble with convection ON using those predictors (one Picard step)
opts = struct();
opts.useConv = (Re ~= 0);   % turn on only if Re>0
opts.au = au_cell;          % cell{e}(Qv x 1)
opts.av = av_cell;          % cell{e}(Qv x 1)
opts.ngauss_v = ngauss_v;   % ensure builder uses same rules
opts.ngauss_p = ngauss_p;

[K,F,meta] = stokes_build_stiffness_Q10Q9(meshVel, meshPres, mu, Re, opts);

% Dirichlet BCs are enforced inside K (as before). Solve:
U = K \ F;

% split
Ux = U(1:NV);
Uy = U(NV+1:2*NV);
P  = U(2*NV+1:end);

%% L2 errors (reuse meta.quadV mapping if builder exposes it; else recompute)
L2u2=0; L2v2=0; L2p2=0;
for e = 1:eV.NE
    idsV = eV.quads(e,:).';
    xe  = nV.x(idsV);  ye = nV.y(idsV);
    xmin=min(xe); xmax=max(xe); dx=xmax-xmin;
    ymin=min(ye); ymax=max(ye); dy=ymax-ymin; J = dx*dy;

    % same perm + Phi as above
    xi_n  = (xe - xmin)/max(dx,eps);
    eta_n = (ye - ymin)/max(dy,eps);
    Lx = bary_eval_matrix(nodes1D, w1, xi_n.');
    Ly = bary_eval_matrix(nodes1D, w1, eta_n.');
    V  = zeros(NlocV,NlocV);
    for m=1:NlocV
        i = IJ(m,1); j = IJ(m,2);
        V(:,m) = (Lx(i,:).').*(Ly(j,:).');
    end
    [~, rows] = max(V,[],1);
    perm_g2t = zeros(NlocV,1); for m=1:NlocV, perm_g2t(rows(m)) = m; end
    PhiV = PhiVt(perm_g2t,:);

    xq = PhiV.' * xe;  yq = PhiV.' * ye;
    uh = PhiV.' * Ux(idsV);
    vh = PhiV.' * Uy(idsV);
    ue = u_ex(xq,yq);  ve = v_ex(xq,yq);

    L2u2 = L2u2 + sum((uh-ue).^2 .* (wV*J));
    L2v2 = L2v2 + sum((vh-ve).^2 .* (wV*J));
end

% pressure L2 on pressure mesh (same idea but with Q9 cache)
NlocP = eP.nloc;
nodes1D_p = linspace(0,1,10).'; w1p = bary_weights(nodes1D_p);

Qp  = ref_quad_cache_p9('ngauss', ngauss_p);
PhiPt = Qp.Phi_ref.';  wp = Qp.Wts(:);
IJp = zeros(NlocP,2); c=0; for j=1:10, for i=1:10, c=c+1; IJp(c,:)=[i,j]; end, end

for e = 1:eP.NE
    idsP = eP.quads(e,:).';
    xe = nP.x(idsP); ye = nP.y(idsP);
    xmin=min(xe); xmax=max(xe); dx=xmax-xmin;
    ymin=min(ye); ymax=max(ye); dy=ymax-ymin; J=dx*dy;

    xi_n  = (xe - xmin)/max(dx,eps);
    eta_n = (ye - ymin)/max(dy,eps);
    Lx = bary_eval_matrix(nodes1D_p, w1p, xi_n.');
    Ly = bary_eval_matrix(nodes1D_p, w1p, eta_n.');
    V  = zeros(NlocP,NlocP);
    for m=1:NlocP
        i = IJp(m,1); j = IJp(m,2);
        V(:,m) = (Lx(i,:).').*(Ly(j,:).');
    end
    [~, rows] = max(V,[],1);
    perm_g2t = zeros(NlocP,1); for m=1:NlocP, perm_g2t(rows(m)) = m; end
    PhiP = PhiPt(perm_g2t,:);

    xq = PhiP.' * xe;  yq = PhiP.' * ye;
    ph = PhiP.' * P(idsP);
    pe = p_ex(xq,yq);

    L2p2 = L2p2 + sum((ph - pe).^2 .* (wp*J));
end

L2u = sqrt(L2u2); L2v = sqrt(L2v2); L2p = sqrt(L2p2);
% “h” just for logging (uniform grid assumed)
hx = max(nV.x) - min(nV.x); ny = sqrt(NV) - 1; h = hx/max(ny,1);

fprintf('\n-- NS MMS (Picard, conv ON) Q10–Q9: %s / %s, mu=%g, Re=%g, GL %dx%d --\n', ...
    meshVel, meshPres, mu, Re, ngauss_v, ngauss_v);
fprintf('L2(u)=%.3e, L2(v)=%.3e, L2(p)=%.3e | log(h)=%.6f\n', L2u, L2v, L2p, log(h));

if show_plots
    figure; subplot(1,2,1); scatter3(nV.x,nV.y, u_ex(nV.x,nV.y), 10, u_ex(nV.x,nV.y),'filled'); title('u exact');
            subplot(1,2,2); scatter3(nV.x,nV.y, Ux, 10, Ux,'filled'); title('u_h');

    figure; subplot(1,2,1); scatter3(nV.x,nV.y, v_ex(nV.x,nV.y), 10, v_ex(nV.x,nV.y),'filled'); title('v exact');
            subplot(1,2,2); scatter3(nV.x,nV.y, Uy, 10, Uy,'filled'); title('v_h');

    figure; subplot(1,2,1); scatter3(nP.x,nP.y, p_ex(nP.x,nP.y), 10, p_ex(nP.x,nP.y),'filled'); title('p exact');
            subplot(1,2,2); scatter3(nP.x,nP.y, P, 10, P,'filled'); title('p_h');
end

% ---- helpers (local) ----
function w=bary_weights(x)
x=x(:); n=numel(x); w=ones(n,1);
for j=1:n, for k=[1:j-1,j+1:n], w(j)=w(j)*(x(j)-x(k)); end, end
w=1./w;
end
function L=bary_eval_matrix(xn,w,xq)
xn=xn(:); xq=xq(:).'; L=zeros(numel(xn),numel(xq));
for k=1:numel(xq)
    x=xq(k); d=x-xn; hit=find(abs(d)<1e-14,1);
    if ~isempty(hit), e=zeros(numel(xn),1); e(hit)=1; L(:,k)=e;
    else, t=w./d; L(:,k)=t/sum(t);
    end
end
end
