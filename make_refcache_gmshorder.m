function make_refcache_gmshorder(p, ngauss, ref_mesh)
% Build a master reference-element cache in **Gmsh local order** once.
% Uses a single reference mesh (e.g. 'squaretest10.m') just to learn the
% Gmsh local ordering; the saved cache is mesh-independent thereafter.
%
% Output: ref/refcache_gmsh_p<p>_g<ngauss>.mat
%   Pts(Qx2), Wts(Qx1)         : Gauss points/weights on [0,1]^2
%   PhiG(QxN2)                 : basis in **Gmsh local order**
%   dPhiG_dxi, dPhiG_deta      : reference derivatives (Gmsh order)
%   N2, p, ngauss, perm_t2g    : info + tensor->gmsh permutation (1xN2)

if nargin<2||isempty(ngauss), ngauss=11; end
if nargin<3||isempty(ref_mesh), ref_mesh='squaretest10.m'; end

% 1) Build (or reuse) tensor-order reference cache
ref_tensor = fullfile('ref', sprintf('refcache_p%d_g%d.mat', p, ngauss));
if ~exist(ref_tensor,'file')
    build_master_refcache(p, ngauss);   % creates ref/refcache_p<p>_g<g>.mat
end
R = load(ref_tensor);  % PhiT (QxN2), dPhiT_dxi, dPhiT_deta, Pts, Wts, N2

% 2) Read the reference mesh and get one element's local ordering
msh = gmshm_to_struct(ref_mesh);
quads_field = find_quads_field_for_p(msh, p);
if isempty(quads_field), error('Ref mesh %s has no QUADS%d.', ref_mesh, (p+1)^2); end
E = msh.(quads_field)(:, 1:R.N2);
ids = E(1,:).';                             % take first element
XY  = msh.POS(ids,1:2);                     % its node coords

% 3) Build permutation gmsh <-> tensor for that element
perm_g2t = gmsh_to_tensor_perm_generic(XY, p);  % gmsh->tensor
perm_t2g = zeros(1,R.N2); perm_t2g(perm_g2t) = 1:R.N2;

% 4) Convert tensor reference arrays into **Gmsh order** once
PhiG       = R.PhiT(:,      perm_t2g);
dPhiG_dxi  = R.dPhiT_dxi(:, perm_t2g);
dPhiG_deta = R.dPhiT_deta(:,perm_t2g);

% 5) Save master **Gmsh-ordered** cache
outdir='ref'; if ~exist(outdir,'dir'), mkdir(outdir); end
outname = fullfile(outdir, sprintf('refcache_gmsh_p%d_g%d.mat', p, ngauss));
Pts=R.Pts; Wts=R.Wts; N2=R.N2; %#ok<NASGU>
save(outname,'Pts','Wts','PhiG','dPhiG_dxi','dPhiG_deta','N2','p','ngauss','perm_t2g');
fprintf('saved %s (p=%d, N2=%d, Q=%d)\n', outname, p, N2, numel(Wts));
end

% ---------- helpers (local, minimal) ----------
function msh = gmshm_to_struct(gmshMfile)
run(gmshMfile);
S = struct();
if exist('msh','var')&&isstruct(msh)
    f=fieldnames(msh);
    for k=1:numel(f), v=msh.(f{k}); if isnumeric(v), S.(f{k})=double(v); end, end
else
    if ~exist('POS','var'), error('POS missing'); end
    S.POS=double(POS); vars=whos;
    for v=vars(:).', nm=v.name
        if startsWith(nm,'QUADS','IgnoreCase',true)||startsWith(nm,'LINES','IgnoreCase',true)
            S.(nm)=double(eval(nm)); %#ok<EVLDIR>
        end
    end
end
S.nbNod=size(S.POS,1); msh=S;
end

function quads_field = find_quads_field_for_p(msh, p)
if p==1
    if isfield(msh,'QUADS4'), quads_field='QUADS4'; return; end
    if isfield(msh,'QUADS'),  quads_field='QUADS';  return; end
end
cand=sprintf('QUADS%d',(p+1)^2);
if isfield(msh,cand), quads_field=cand; else, quads_field=''; end
end

function perm_g2t = gmsh_to_tensor_perm_generic(XY, p)
N1=p+1; N2=N1*N1;
xmin=min(XY(:,1)); xmax=max(XY(:,1)); dx=xmax-xmin;
ymin=min(XY(:,2)); ymax=max(XY(:,2)); dy=ymax-ymin;
xi=(XY(:,1)-xmin)/dx; eta=(XY(:,2)-ymin)/dy;
r1=linspace(0,1,N1).'; w1=bary_weights(r1);
Lx=bary_eval_matrix(r1,w1,xi.'); Ly=bary_eval_matrix(r1,w1,eta.');
IJ=zeros(N2,2); c=0; for j=1:N1, for i=1:N1, c=c+1; IJ(c,:)=[i,j]; end, end
V=zeros(N2,N2);
for m=1:N2, i=IJ(m,1); j=IJ(m,2); V(:,m)=(Lx(i,:).').*(Ly(j,:).'); end
[~,rows]=max(V,[],1); perm_g2t=zeros(N2,1); for m=1:N2, perm_g2t(rows(m))=m; end
end
function w=bary_weights(x)
x=x(:); n=numel(x); w=ones(n,1);
for j=1:n, for k=[1:j-1,j+1:n], w(j)=w(j)*(x(j)-x(k)); end, end, w=1./w;
end
function L=bary_eval_matrix(xnodes,w,xq)
xn=xnodes(:); xq=xq(:).'; L=zeros(numel(xn),numel(xq));
for k=1:numel(xq)
    x=xq(k); d=x-xn; hit=find(abs(d)<1e-14,1);
    if ~isempty(hit), e=zeros(numel(xn),1); e(hit)=1; L(:,k)=e;
    else, t=w./d; L(:,k)=t/sum(t);
    end
end
end
